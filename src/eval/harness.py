"""
Evaluation harness: runs the agent against eval/holdout scenarios,
computes KPIs, and produces before/after comparison reports.
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from src.agent.react_agent import ReActAgent, LLMBackend
from src.environment.sqli_env import SQLiEnvironment
from src.environment.scenarios import (
    Scenario,
    get_eval_scenarios,
    get_holdout_scenarios,
    get_train_scenarios,
)
from src.eval.metrics import (
    EpisodeKPIs,
    RegressionReport,
    aggregate_kpis,
    compute_afs,
    detect_regressions,
)
from src.rl.policy import UCBBandit

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Eval config
# ---------------------------------------------------------------------------

@dataclass
class EvalConfig:
    runs_per_scenario: int = 3
    split: str = "eval"              # eval | holdout | train | all
    strategy_override: Optional[str] = None
    bandit_path: Optional[str] = None
    save_dir: str = "results/eval"
    llm_provider: str = "mock"
    llm_model: Optional[str] = None
    app_port: int = 5001
    verbose: bool = False

    def to_dict(self) -> Dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Eval report
# ---------------------------------------------------------------------------

@dataclass
class EvalReport:
    run_id: str
    config: Dict
    scenario_results: Dict[str, List[Dict]] = field(default_factory=dict)
    scenario_aggregates: Dict[str, Dict] = field(default_factory=dict)
    overall_aggregate: Dict = field(default_factory=dict)
    afs: float = 0.0
    fdr_on_clean: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return {
            "run_id": self.run_id,
            "config": self.config,
            "scenario_results": self.scenario_results,
            "scenario_aggregates": self.scenario_aggregates,
            "overall_aggregate": self.overall_aggregate,
            "afs": self.afs,
            "fdr_on_clean": self.fdr_on_clean,
            "timestamp": self.timestamp,
        }

    def save(self, save_dir: str) -> str:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"eval_{self.run_id}.json")
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path

    def print_summary(self):
        """Print a human-readable summary to stdout."""
        print("\n" + "=" * 70)
        print(f"EVAL REPORT  {self.run_id}")
        print("=" * 70)
        print(f"Agent Fitness Score (AFS):  {self.afs:.4f}")
        print(f"False Discovery Rate (clean): {self.fdr_on_clean:.4f}")
        print()

        for sc_id, agg in self.scenario_aggregates.items():
            dcs = agg.get("dcs", {})
            cs = agg.get("composite_score", {})
            fp = agg.get("fp_count_total", 0)
            print(
                f"  [{sc_id}]  DCS={dcs.get('mean', 0):.3f}±{dcs.get('std', 0):.3f}  "
                f"Composite={cs.get('mean', 0):.3f}  FP={fp}"
            )

        print()
        overall_dcs = self.overall_aggregate.get("dcs", {}).get("mean", 0)
        overall_cs = self.overall_aggregate.get("composite_score", {}).get("mean", 0)
        print(f"Overall DCS (mean):        {overall_dcs:.4f}")
        print(f"Overall Composite (mean):  {overall_cs:.4f}")
        print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

class EvalHarness:
    """
    Runs agent episodes on eval/holdout scenarios and computes KPIs.

    Usage:
        harness = EvalHarness(config)
        report = harness.run()
        report.print_summary()
        report.save("results/")
    """

    def __init__(self, config: EvalConfig):
        self.config = config
        self._run_id = f"eval_{int(time.time())}"

        self.env = SQLiEnvironment(port=config.app_port)
        self.llm = LLMBackend(provider=config.llm_provider, model=config.llm_model)
        self.agent = ReActAgent(llm=self.llm)

        self.bandit: Optional[UCBBandit] = None
        if config.bandit_path and os.path.isfile(config.bandit_path):
            self.bandit = UCBBandit.load(config.bandit_path)
            logger.info(f"Loaded bandit from {config.bandit_path}")

        self._scenarios = self._load_scenarios()

    def _load_scenarios(self) -> List[Scenario]:
        split = self.config.split
        if split == "eval":
            return get_eval_scenarios()
        elif split == "holdout":
            return get_holdout_scenarios()
        elif split == "train":
            return get_train_scenarios()
        else:
            return get_train_scenarios() + get_eval_scenarios() + get_holdout_scenarios()

    def run(self) -> EvalReport:
        """Execute the full evaluation and return a report."""
        self.env.start_server()
        report = EvalReport(run_id=self._run_id, config=self.config.to_dict())

        all_kpis: List[EpisodeKPIs] = []

        for scenario in self._scenarios:
            strategy = self._select_strategy(scenario.scenario_id)
            logger.info(
                f"Evaluating {scenario.scenario_id} with strategy={strategy} "
                f"× {self.config.runs_per_scenario} runs"
            )

            scenario_kpis: List[EpisodeKPIs] = []
            episode_dicts: List[Dict] = []

            for run_n in range(self.config.runs_per_scenario):
                self.agent.reset(strategy_id=strategy)
                metrics, trajectory = self.agent.run_episode(
                    env=self.env, scenario=scenario, verbose=self.config.verbose
                )

                kpi = EpisodeKPIs.from_metrics_dict(metrics, scenario.scenario_id, strategy)
                scenario_kpis.append(kpi)
                all_kpis.append(kpi)

                ep_dict = kpi.to_dict()
                ep_dict["run_n"] = run_n
                ep_dict["trajectory_len"] = len(trajectory)
                episode_dicts.append(ep_dict)

                if self.config.verbose:
                    logger.info(
                        f"  run {run_n+1}: DCS={kpi.dcs:.3f} composite={kpi.composite_score():.3f} "
                        f"TP={kpi.true_positives} FP={kpi.false_positives}"
                    )

            agg = aggregate_kpis(scenario_kpis)
            report.scenario_results[scenario.scenario_id] = episode_dicts
            report.scenario_aggregates[scenario.scenario_id] = agg

        report.overall_aggregate = aggregate_kpis(all_kpis)

        scenario_cs = {
            sc_id: agg.get("composite_score", {}).get("mean", 0.0)
            for sc_id, agg in report.scenario_aggregates.items()
        }
        report.afs = compute_afs(scenario_cs)

        holdout_clean_kpis = [k for k in all_kpis if "holdout_clean" in k.scenario_id]
        if holdout_clean_kpis:
            total_reports = sum(k.true_positives + k.false_positives for k in holdout_clean_kpis)
            false_reports = sum(k.false_positives for k in holdout_clean_kpis)
            report.fdr_on_clean = round(false_reports / max(total_reports, 1), 4)

        path = report.save(self.config.save_dir)
        logger.info(f"Eval report saved to {path}")

        return report

    def _select_strategy(self, scenario_id: str) -> str:
        """Select strategy: override > bandit best arm > default."""
        if self.config.strategy_override:
            return self.config.strategy_override
        if self.bandit:
            best_arm, _ = self.bandit.best_arm(scenario_id)
            return best_arm
        from src.agent.prompts import DEFAULT_STRATEGY
        return DEFAULT_STRATEGY


# ---------------------------------------------------------------------------
# Before/After comparison
# ---------------------------------------------------------------------------

def compare_eval_reports(
    before_path: str,
    after_path: str,
    regression_threshold: float = 0.05,
) -> RegressionReport:
    """
    Load two eval reports and produce a regression/improvement comparison.

    This is the core of the regression detection system described in
    kpi_framework.md — run after any base LLM update or prompt change.
    """
    with open(before_path) as f:
        before = json.load(f)
    with open(after_path) as f:
        after = json.load(f)

    before_agg = before.get("overall_aggregate", {})
    after_agg = after.get("overall_aggregate", {})

    report = detect_regressions(
        baseline=before_agg,
        current=after_agg,
        baseline_run_id=before.get("run_id", "before"),
        current_run_id=after.get("run_id", "after"),
        regression_threshold=regression_threshold,
    )

    report.afs_delta = round(
        after.get("afs", 0.0) - before.get("afs", 0.0), 4
    )

    return report


def print_comparison(before_path: str, after_path: str):
    """Pretty-print a before/after comparison."""
    report = compare_eval_reports(before_path, after_path)

    print("\n" + "=" * 70)
    print(f"REGRESSION CHECK: {report.baseline_run_id} → {report.current_run_id}")
    print("=" * 70)
    print(f"AFS delta: {report.afs_delta:+.4f}")

    if report.improvements:
        print("\n✅ IMPROVEMENTS:")
        for item in report.improvements:
            print(f"  {item['metric']}: {item['baseline']:.4f} → {item['current']:.4f} ({item['delta']:+.4f})")

    if report.regressions:
        print("\n❌ REGRESSIONS:")
        for item in report.regressions:
            sev = item.get("severity", "warning").upper()
            print(f"  [{sev}] {item['metric']}: {item['baseline']:.4f} → {item['current']:.4f} ({item['delta']:+.4f})")

    if report.stable:
        print(f"\n➡  STABLE: {', '.join(report.stable)}")

    print(f"\nOverall: {'⚠️  REGRESSION DETECTED' if report.has_regression else '✅ NO REGRESSION'}")
    print("=" * 70 + "\n")
