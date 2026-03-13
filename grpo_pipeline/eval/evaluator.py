"""
Full KPI evaluator for the GRPO pipeline.

Tracks ALL metrics from kpi_framework.md:
  Effectiveness:  DCS, True Positives, False Positives
  Efficiency:     STC Score, TER (Token Efficiency Ratio), TIE
  Autonomy:       HIR (always 0 — fully automated), DERR, STA
  Diversity:      AVDS
  Composite:      AFS (Agent Fitness Score)

Produces an EvalSnapshot per run and a before/after comparison table.
"""

import os
import sys
import json
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.agent.prompts import build_system_prompt, build_task_prompt, DEFAULT_STRATEGY
from src.agent.tools import parse_llm_output
from src.environment.sqli_env import SQLiEnvironment
from src.environment.scenarios import get_eval_scenarios
from grpo_pipeline.agent.grpo_prompts import build_grpo_system_prompt


# ------------------------------------------------------------------
# KPI snapshot dataclass
# ------------------------------------------------------------------

@dataclass
class KPISnapshot:
    label: str                          # e.g. "pre_grpo" / "post_grpo_epoch_5"

    # Effectiveness
    dcs: float = 0.0                    # Discovery Coverage Score
    true_positives: float = 0.0         # mean TPs per episode
    false_positives: float = 0.0        # mean FPs per episode

    # Efficiency
    stc_score: float = 0.0              # Steps-to-Confirmation score
    ter: float = 0.0                    # Token Efficiency Ratio = DCS / (tokens/1K)
    tie: float = 0.0                    # Tool Invocation Efficiency

    # Autonomy
    hir: float = 0.0                    # Human Intervention Rate (always 0 — fully automated)
    derr: float = 0.0                   # Dead-End Recovery Rate
    sta: float = 0.0                    # Self-Termination Accuracy

    # Diversity
    avds: float = 0.0                   # Attack Vector Diversity Score

    # Composite
    afs: float = 0.0                    # Agent Fitness Score

    # RL signal
    mean_reward: float = 0.0
    total_episodes: int = 0

    def to_dict(self) -> Dict:
        return asdict(self)


# ------------------------------------------------------------------
# Per-episode metric computation
# ------------------------------------------------------------------

def _compute_episode_kpis(
    state,
    scenario: Any,
    steps_taken: int,
    tokens_used: int,
    total_tool_calls: int,
    productive_tool_calls: int,
    dead_end_recoveries: int,
    dead_ends_encountered: int,
    termination: str,
) -> Dict:
    """Compute all KPIs for a single episode."""
    known_vulns = len(scenario.active_vulns)
    vuln_keys = {(v.endpoint, v.param) for v in scenario.active_vulns}
    tp = sum(1 for f in state.findings if (f.endpoint, f.param) in vuln_keys
             or any(f.endpoint.endswith(v.endpoint.split("{")[0].rstrip("/")) for v in scenario.active_vulns))
    fp = len(state.findings) - tp

    # DCS
    dcs = max(0.0, (tp / known_vulns) - 2.0 * (fp / max(len(state.findings), 1))) if known_vulns > 0 else 0.0
    dcs = min(dcs, 1.0)

    # STC score (optimal = 2*known_vulns steps: enumerate + inject each)
    steps_optimal = 1 + 2 * known_vulns
    steps_budget = scenario.step_budget
    stc = max(0.0, 1.0 - (steps_taken - steps_optimal) / max(steps_budget - steps_optimal, 1))

    # TER
    ter = dcs / (tokens_used / 1000.0) if tokens_used > 0 else 0.0

    # TIE
    tie = productive_tool_calls / max(total_tool_calls, 1)

    # HIR — always 0 (fully automated, no human interventions)
    hir = 0.0

    # DERR
    derr = dead_end_recoveries / max(dead_ends_encountered, 1) if dead_ends_encountered > 0 else 1.0

    # STA — 1.0 if agent stopped cleanly with correct DCS, 0.0 otherwise
    premature = 1 if (termination == "agent_stopped" and dcs < 0.8 and known_vulns > 0) else 0
    overrun = 1 if termination == "budget" and dcs >= 1.0 else 0
    sta = 1.0 - premature - overrun
    sta = max(0.0, sta)

    # AVDS
    attack_types_used = set(f.vuln_type for f in state.findings)
    applicable_types = {"error_based", "union_based", "blind_boolean"}
    avds = len(attack_types_used & applicable_types) / len(applicable_types)

    # AFS composite: 0.4*DCS + 0.25*STC + 0.2*(1-HIR) + 0.15*AVDS
    afs = 0.4 * dcs + 0.25 * stc + 0.2 * (1.0 - hir) + 0.15 * avds

    return {
        "dcs": dcs, "tp": tp, "fp": fp,
        "stc": stc, "ter": ter, "tie": tie,
        "hir": hir, "derr": derr, "sta": sta,
        "avds": avds, "afs": afs,
    }


# ------------------------------------------------------------------
# Episode runner
# ------------------------------------------------------------------

def _run_eval_episode(llm, env: SQLiEnvironment, scenario: Any, max_steps: int, verbose: bool = False) -> Dict:
    """Run one eval episode and return all KPI inputs."""
    system_prompt = build_grpo_system_prompt(build_system_prompt(DEFAULT_STRATEGY))
    state = env.reset(scenario)

    total_tool_calls = 0
    productive_tool_calls = 0
    tokens_used = 0
    dead_ends = 0
    dead_end_recoveries = 0
    last_dead_end_streak = 0
    reward_sum = 0.0
    termination = "budget"

    for step in range(max_steps):
        user_prompt = build_task_prompt(
            state_context=state.to_prompt_context(),
            scenario_description=scenario.description,
        )
        completion_text, completion_ids = llm.generate(system_prompt, user_prompt)
        tokens_used += len(completion_ids)

        action = parse_llm_output(completion_text)
        if not action.is_valid:
            action.tool = "stop"
            action.params = {"reason": "parse_error"}

        if step == 0 and action.tool != "enumerate_endpoints":
            action.tool = "enumerate_endpoints"
            action.params = {}
        elif step < 2 and action.tool == "stop":
            action.tool = "enumerate_endpoints"
            action.params = {}

        if verbose:
            ep = action.params.get("endpoint", action.params.get("url", ""))
            param = action.params.get("param", "")
            payload = action.params.get("payload", "")
            method = action.params.get("method", "GET")
            reason = action.params.get("reason", "")
            sep = "-" * 60
            print(sep)
            if action.tool in ("inject_payload", "http_request"):
                print(f"  ▶ STEP {step} | {action.tool}")
                print(f"    REQUEST : {method} {ep}")
                print(f"    PARAM   : {param!r}")
                print(f"    PAYLOAD : {payload!r}")
            elif action.tool == "report_finding":
                print(f"  ▶ STEP {step} | report_finding")
                print(f"    ENDPOINT: {ep}  PARAM={param!r}")
                print(f"    TYPE    : {action.params.get('vuln_type','?')}  CONF={action.params.get('confidence','?')}")
                print(f"    EVIDENCE: {action.params.get('evidence','')[:120]!r}")
            elif action.tool == "stop":
                print(f"  ▶ STEP {step} | stop  reason={reason!r}")
            else:
                print(f"  ▶ STEP {step} | {action.tool}")

        prev_findings = len(state.findings)
        prev_endpoints = len(state.discovered_endpoints)

        next_state, reward, done, _info = env.step(action.to_env_action())
        reward_sum += reward
        total_tool_calls += 1

        if verbose:
            result = _info.get("result", {})
            status = result.get("status_code", "n/a")
            body = result.get("body", "")
            signals = result.get("injection_signals", [])
            error = result.get("error", "")
            print(f"    RESPONSE: status={status}")
            if signals:
                print(f"    SIGNALS : {signals}  ← injection detected!")
            if error:
                print(f"    ERROR   : {error!r}")
            if body:
                snippet = body[:200].replace("\n", " ")
                print(f"    BODY    : {snippet!r}")
            print(f"    REWARD  : {reward:+.2f}  |  findings so far: {len(next_state.findings)}")

        # Productive = call that changed state or returned signals
        new_findings = len(next_state.findings) > prev_findings
        new_endpoints = len(next_state.discovered_endpoints) > prev_endpoints
        has_signals = bool(_info.get("result", {}).get("injection_signals"))
        if new_findings or new_endpoints or has_signals:
            productive_tool_calls += 1

        # Dead-end tracking
        curr_streak = next_state.dead_end_streak
        if curr_streak > last_dead_end_streak and curr_streak >= 3:
            dead_ends += 1
        elif last_dead_end_streak >= 3 and curr_streak < last_dead_end_streak:
            dead_end_recoveries += 1
        last_dead_end_streak = curr_streak

        state = next_state

        if done:
            termination = "agent_stopped" if action.tool == "stop" else "budget"
            break

    kpis = _compute_episode_kpis(
        state=state,
        scenario=scenario,
        steps_taken=step + 1,
        tokens_used=tokens_used,
        total_tool_calls=total_tool_calls,
        productive_tool_calls=productive_tool_calls,
        dead_end_recoveries=dead_end_recoveries,
        dead_ends_encountered=dead_ends,
        termination=termination,
    )
    kpis["reward"] = reward_sum
    return kpis


# ------------------------------------------------------------------
# Main evaluator
# ------------------------------------------------------------------

class GRPOEvaluator:
    """
    Runs N eval episodes across all eval scenarios and aggregates all KPIs.
    Call evaluate() before and after GRPO training to produce a comparison.
    """

    def __init__(self, llm, env: SQLiEnvironment, episodes_per_scenario: int = 3):
        self.llm = llm
        self.env = env
        self.eps = episodes_per_scenario

    def evaluate(self, label: str, verbose: bool = False) -> KPISnapshot:
        """Run full evaluation and return a KPISnapshot."""
        scenarios = get_eval_scenarios()
        all_kpis: List[Dict] = []

        print(f"\n[Evaluator] Running eval: {label}")
        for scenario in scenarios:
            for ep_idx in range(self.eps):
                if verbose:
                    print(f"  Scenario: {scenario.name} | episode {ep_idx+1}/{self.eps}")
                kpis = _run_eval_episode(
                    self.llm, self.env, scenario, scenario.step_budget, verbose=verbose
                )
                all_kpis.append(kpis)

        n = max(len(all_kpis), 1)

        def mean(key):
            return sum(k[key] for k in all_kpis) / n

        snap = KPISnapshot(
            label=label,
            dcs=mean("dcs"),
            true_positives=mean("tp"),
            false_positives=mean("fp"),
            stc_score=mean("stc"),
            ter=mean("ter"),
            tie=mean("tie"),
            hir=mean("hir"),
            derr=mean("derr"),
            sta=mean("sta"),
            avds=mean("avds"),
            afs=mean("afs"),
            mean_reward=mean("reward"),
            total_episodes=n,
        )
        print(f"[Evaluator] {label}: DCS={snap.dcs:.3f}  AFS={snap.afs:.3f}  "
              f"TP={snap.true_positives:.1f}  FP={snap.false_positives:.1f}")
        return snap


# ------------------------------------------------------------------
# Before / after comparison printer
# ------------------------------------------------------------------

def print_comparison(before: KPISnapshot, after: KPISnapshot) -> None:
    """Print a rich before/after table for all KPIs."""
    try:
        from rich.console import Console
        from rich.table import Table
        console = Console()
        table = Table(title="GRPO Before → After — Full KPI Comparison", show_lines=True)
        table.add_column("KPI", style="cyan")
        table.add_column("Category", style="dim")
        table.add_column(f"Before ({before.label})", justify="right")
        table.add_column(f"After ({after.label})", justify="right")
        table.add_column("Delta", justify="right")
        table.add_column("Target", justify="right")
        table.add_column("Status")

        rows = [
            ("DCS", "Effectiveness", before.dcs, after.dcs, "≥ 0.75"),
            ("True Positives", "Effectiveness", before.true_positives, after.true_positives, "max"),
            ("False Positives", "Effectiveness", before.false_positives, after.false_positives, "= 0"),
            ("STC Score", "Efficiency", before.stc_score, after.stc_score, "≥ 0.60"),
            ("TER", "Efficiency", before.ter, after.ter, "≥ 0.05"),
            ("TIE", "Efficiency", before.tie, after.tie, "≥ 0.60"),
            ("HIR", "Autonomy", before.hir, after.hir, "≤ 0.05"),
            ("DERR", "Autonomy", before.derr, after.derr, "≥ 0.70"),
            ("STA", "Autonomy", before.sta, after.sta, "≥ 0.80"),
            ("AVDS", "Diversity", before.avds, after.avds, "≥ 0.50"),
            ("AFS", "Composite", before.afs, after.afs, "≥ 0.60"),
            ("Mean Reward", "RL Signal", before.mean_reward, after.mean_reward, "max"),
        ]

        for name, category, b, a, target in rows:
            delta = a - b
            sign = "+" if delta >= 0 else ""
            is_fp = name == "False Positives"
            improved = delta < 0 if is_fp else delta > 0
            status = "✅ improved" if improved else ("➡ stable" if abs(delta) < 0.01 else "⚠ regressed")
            table.add_row(
                name, category,
                f"{b:.3f}", f"{a:.3f}",
                f"[green]{sign}{delta:.3f}[/green]" if improved else f"{sign}{delta:.3f}",
                target, status,
            )

        console.print(table)

    except ImportError:
        print("\n=== GRPO Before → After — Full KPI Comparison ===")
        print(f"{'KPI':<20} {'Before':>8} {'After':>8} {'Delta':>8}")
        print("-" * 50)
        pairs = [
            ("DCS", before.dcs, after.dcs),
            ("True Positives", before.true_positives, after.true_positives),
            ("False Positives", before.false_positives, after.false_positives),
            ("STC Score", before.stc_score, after.stc_score),
            ("TER", before.ter, after.ter),
            ("TIE", before.tie, after.tie),
            ("HIR", before.hir, after.hir),
            ("DERR", before.derr, after.derr),
            ("STA", before.sta, after.sta),
            ("AVDS", before.avds, after.avds),
            ("AFS", before.afs, after.afs),
            ("Mean Reward", before.mean_reward, after.mean_reward),
        ]
        for name, b, a in pairs:
            delta = a - b
            print(f"{name:<20} {b:>8.3f} {a:>8.3f} {delta:>+8.3f}")


def save_snapshot(snap: KPISnapshot, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(snap.to_dict(), f, indent=2)
    print(f"[Evaluator] Snapshot saved → {path}")


def load_snapshot(path: str) -> KPISnapshot:
    with open(path) as f:
        data = json.load(f)
    return KPISnapshot(**data)
