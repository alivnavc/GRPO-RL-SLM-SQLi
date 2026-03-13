"""
KPI computation and aggregation for the eval harness.

Implements the formulas from kpi_framework.md for batch evaluation.
"""

import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Per-episode KPI computation (wraps env.get_episode_metrics)
# ---------------------------------------------------------------------------

@dataclass
class EpisodeKPIs:
    scenario_id: str
    strategy_id: str
    dcs: float              # Discovery Coverage Score
    stc_score: float        # Steps-to-Confirmation Score
    ter: float              # Token Efficiency Ratio
    avds: float             # Attack Vector Diversity Score
    hir: float              # Human Intervention Rate
    derr: float             # Dead-End Recovery Rate
    tie: float              # Tool Invocation Efficiency
    true_positives: int
    false_positives: int
    known_vulns: int
    steps_taken: int
    tokens_used: int
    wall_time_s: float
    total_reward: float
    termination_reason: str

    @classmethod
    def from_metrics_dict(
        cls, metrics: Dict[str, Any], scenario_id: str, strategy_id: str
    ) -> "EpisodeKPIs":
        return cls(
            scenario_id=scenario_id,
            strategy_id=strategy_id,
            dcs=metrics.get("dcs", 0.0),
            stc_score=metrics.get("stc_score", 0.0),
            ter=metrics.get("ter", 0.0),
            avds=metrics.get("avds", 0.0),
            hir=metrics.get("hir", 0.0),
            derr=metrics.get("derr", 1.0),
            tie=metrics.get("tie", 0.0),
            true_positives=metrics.get("true_positives", 0),
            false_positives=metrics.get("false_positives", 0),
            known_vulns=metrics.get("known_vulns", 0),
            steps_taken=metrics.get("steps_taken", 0),
            tokens_used=metrics.get("tokens_used", 0),
            wall_time_s=metrics.get("wall_time_s", 0.0),
            total_reward=metrics.get("total_reward", 0.0),
            termination_reason=metrics.get("termination_reason", ""),
        )

    def composite_score(self) -> float:
        """
        composite_task_score from kpi_framework.md:
        0.4*DCS + 0.25*STC + 0.2*autonomy_proxy + 0.15*reasoning_proxy

        autonomy_proxy = (1 - HIR) * DERR
        reasoning_proxy = AVDS * TIE
        """
        autonomy = (1 - self.hir) * self.derr
        reasoning = self.avds * max(self.tie, 0.1)
        return round(
            0.40 * self.dcs
            + 0.25 * self.stc_score
            + 0.20 * autonomy
            + 0.15 * reasoning,
            4,
        )

    def meets_thresholds(self) -> Dict[str, bool]:
        """Check each KPI against the targets from kpi_framework.md."""
        return {
            "dcs_ok": self.dcs >= 0.75,
            "stc_ok": self.stc_score >= 0.60,
            "ter_ok": self.ter >= 0.05,
            "avds_ok": self.avds >= 0.50,
            "hir_ok": self.hir <= 0.05,
            "derr_ok": self.derr >= 0.70,
            "tie_ok": self.tie >= 0.60,
            "fp_zero": self.false_positives == 0,
        }

    def to_dict(self) -> Dict:
        return {
            "scenario_id": self.scenario_id,
            "strategy_id": self.strategy_id,
            "dcs": self.dcs,
            "stc_score": self.stc_score,
            "ter": self.ter,
            "avds": self.avds,
            "hir": self.hir,
            "derr": self.derr,
            "tie": self.tie,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "known_vulns": self.known_vulns,
            "steps_taken": self.steps_taken,
            "tokens_used": self.tokens_used,
            "wall_time_s": self.wall_time_s,
            "total_reward": self.total_reward,
            "termination_reason": self.termination_reason,
            "composite_score": self.composite_score(),
            "meets_thresholds": self.meets_thresholds(),
        }


# ---------------------------------------------------------------------------
# Batch aggregation
# ---------------------------------------------------------------------------

def aggregate_kpis(episodes: List[EpisodeKPIs]) -> Dict[str, Any]:
    """
    Aggregate KPIs across multiple episodes.
    Returns means, stds, and AFS (Agent Fitness Score).
    """
    if not episodes:
        return {}

    def stats(values: List[float]) -> Dict:
        arr = np.array(values)
        return {
            "mean": round(float(np.mean(arr)), 4),
            "std": round(float(np.std(arr)), 4),
            "min": round(float(np.min(arr)), 4),
            "max": round(float(np.max(arr)), 4),
            "median": round(float(np.median(arr)), 4),
        }

    return {
        "n_episodes": len(episodes),
        "dcs": stats([e.dcs for e in episodes]),
        "stc_score": stats([e.stc_score for e in episodes]),
        "ter": stats([e.ter for e in episodes]),
        "avds": stats([e.avds for e in episodes]),
        "hir": stats([e.hir for e in episodes]),
        "derr": stats([e.derr for e in episodes]),
        "tie": stats([e.tie for e in episodes]),
        "composite_score": stats([e.composite_score() for e in episodes]),
        "total_reward": stats([e.total_reward for e in episodes]),
        "tp_rate": round(
            sum(e.true_positives for e in episodes)
            / max(sum(e.known_vulns for e in episodes), 1),
            4,
        ),
        "fp_count_total": sum(e.false_positives for e in episodes),
        "false_discovery_rate": round(
            sum(e.false_positives for e in episodes)
            / max(sum(e.false_positives + e.true_positives for e in episodes), 1),
            4,
        ),
    }


def compute_afs(scenario_scores: Dict[str, float]) -> float:
    """
    Agent Fitness Score (AFS) from kpi_framework.md:
    Weighted mean of per-scenario composite scores.
    """
    task_weights = {
        "train_easy_single": 1.5,
        "train_medium_two": 1.3,
        "train_hard_blind": 1.3,
        "train_full": 1.5,
        "eval_easy": 1.5,
        "eval_medium": 1.3,
        "eval_blind": 1.2,
        "holdout_clean": 1.0,
        "holdout_partial": 1.0,
    }
    total_weight = 0.0
    weighted_sum = 0.0
    for sc_id, score in scenario_scores.items():
        w = task_weights.get(sc_id, 1.0)
        weighted_sum += w * score
        total_weight += w
    return round(weighted_sum / max(total_weight, 1.0), 4)


# ---------------------------------------------------------------------------
# Regression detection
# ---------------------------------------------------------------------------

@dataclass
class RegressionReport:
    """Result of comparing two evaluation runs."""
    baseline_run_id: str
    current_run_id: str
    regressions: List[Dict[str, Any]] = field(default_factory=list)
    improvements: List[Dict[str, Any]] = field(default_factory=list)
    stable: List[str] = field(default_factory=list)
    afs_delta: float = 0.0
    has_regression: bool = False


def detect_regressions(
    baseline: Dict[str, Any],
    current: Dict[str, Any],
    baseline_run_id: str = "baseline",
    current_run_id: str = "current",
    regression_threshold: float = 0.05,
) -> RegressionReport:
    """
    Compare two aggregated evaluation results for regressions.

    A regression is flagged when a metric drops by more than regression_threshold
    (default 5 percentage points) from baseline.

    Key metrics checked:
    - dcs.mean       (primary — must not regress)
    - composite_score.mean
    - fp_count_total (must not increase)
    - false_discovery_rate
    """
    report = RegressionReport(
        baseline_run_id=baseline_run_id,
        current_run_id=current_run_id,
    )

    critical_metrics = [
        ("dcs.mean", "dcs", "mean"),
        ("composite_score.mean", "composite_score", "mean"),
        ("stc_score.mean", "stc_score", "mean"),
        ("avds.mean", "avds", "mean"),
    ]

    for label, key, subkey in critical_metrics:
        base_val = baseline.get(key, {}).get(subkey, 0.0)
        curr_val = current.get(key, {}).get(subkey, 0.0)
        delta = curr_val - base_val

        if delta < -regression_threshold:
            report.regressions.append({
                "metric": label,
                "baseline": base_val,
                "current": curr_val,
                "delta": round(delta, 4),
                "severity": "critical" if key == "dcs" else "warning",
            })
        elif delta > regression_threshold:
            report.improvements.append({
                "metric": label,
                "baseline": base_val,
                "current": curr_val,
                "delta": round(delta, 4),
            })
        else:
            report.stable.append(label)

    base_fp = baseline.get("fp_count_total", 0)
    curr_fp = current.get("fp_count_total", 0)
    if curr_fp > base_fp:
        report.regressions.append({
            "metric": "fp_count_total",
            "baseline": base_fp,
            "current": curr_fp,
            "delta": curr_fp - base_fp,
            "severity": "critical",
        })

    report.has_regression = len(report.regressions) > 0
    report.afs_delta = round(
        current.get("composite_score", {}).get("mean", 0.0)
        - baseline.get("composite_score", {}).get("mean", 0.0),
        4,
    )

    return report
