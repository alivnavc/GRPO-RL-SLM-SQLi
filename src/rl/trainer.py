"""
RL Training loop: UCB Bandit over strategy variants + trajectory logging.

Training approach: Multi-Armed Bandit (UCB1) over prompt strategy variants.

Justification for this approach vs alternatives:
  - PPO over token logits: Requires differentiable policy; impractical for
    black-box LLM APIs. Would require fine-tuning the model weights.
  - RLHF preference learning: Requires human labelers. We automate this via
    KPI-scored trajectories (auto-RLHF).
  - DPO on ranked trajectories: Valid and complementary. We build preference
    pairs in TrajectoryBuffer for potential DPO fine-tuning offline.
  - Bandit over strategy variants: Directly optimizable, interpretable,
    measurably improving (arm concentration shifts), and production-ready
    since strategy selection is just a config change at inference time.

The bandit selects WHICH prompt strategy to use for each episode.
The reward signal comes from the KPI-derived trajectory score.
Over training iterations, the bandit concentrates on the strategy that
achieves the highest DCS + efficiency + autonomy composite.
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.agent.react_agent import ReActAgent, LLMBackend
from src.environment.sqli_env import SQLiEnvironment
from src.environment.scenarios import Scenario, get_train_scenarios
from src.rl.policy import UCBBandit
from src.rl.trajectory_buffer import TrajectoryBuffer
from src.reward.reward_function import score_trajectory

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Training config
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    num_episodes: int = 50
    scenarios: Optional[List[str]] = None    # None = use all train scenarios
    exploration_constant: float = 1.41
    save_dir: str = "results/training"
    checkpoint_every: int = 10
    verbose: bool = False
    llm_provider: str = "mock"
    llm_model: Optional[str] = None
    app_port: int = 5001

    def to_dict(self) -> Dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Training run record
# ---------------------------------------------------------------------------

@dataclass
class TrainingRun:
    run_id: str
    config: Dict
    episode_logs: List[Dict] = field(default_factory=list)
    bandit_snapshots: List[Dict] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None

    def log_episode(
        self,
        episode_num: int,
        scenario_id: str,
        strategy_id: str,
        metrics: Dict,
        score: float,
    ):
        self.episode_logs.append({
            "episode": episode_num,
            "scenario_id": scenario_id,
            "strategy_id": strategy_id,
            "metrics": metrics,
            "score": score,
            "timestamp": time.time(),
        })

    def snapshot_bandit(self, bandit: UCBBandit, episode_num: int):
        snap = {
            "episode": episode_num,
            "convergence": {
                sc: bandit.convergence_summary(sc)
                for sc in set(
                    log["scenario_id"] for log in self.episode_logs
                )
            },
        }
        self.bandit_snapshots.append(snap)

    def to_dict(self) -> Dict:
        return {
            "run_id": self.run_id,
            "config": self.config,
            "episode_logs": self.episode_logs,
            "bandit_snapshots": self.bandit_snapshots,
            "start_time": self.start_time,
            "end_time": self.end_time,
        }

    def save(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"run_{self.run_id}.json")
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class SQLiBanditTrainer:
    """
    Main training loop. Runs episodes, updates bandit, logs results.

    Training loop pseudocode:
        for episode in range(num_episodes):
            scenario = sample_scenario(train_scenarios)
            strategy = bandit.select(scenario.id)      ← UCB1 selection
            metrics, traj = agent.run_episode(env, scenario, strategy)
            score = score_trajectory(metrics, traj, scenario)
            bandit.update(scenario.id, strategy, score) ← update arm
            buffer.add(scenario.id, strategy, metrics, traj, scenario)
            if episode % checkpoint_every == 0:
                save bandit + run log
    """

    def __init__(self, config: TrainingConfig):
        self.config = config
        self._run_id = f"{int(time.time())}"

        self.env = SQLiEnvironment(port=config.app_port)
        self.llm = LLMBackend(provider=config.llm_provider, model=config.llm_model)
        self.agent = ReActAgent(llm=self.llm)
        self.bandit = UCBBandit(exploration_constant=config.exploration_constant)
        self.buffer = TrajectoryBuffer(
            save_dir=os.path.join(config.save_dir, "episodes")
        )

        self._train_scenarios = self._load_scenarios()
        self._rng = np.random.default_rng(seed=42)

        os.makedirs(config.save_dir, exist_ok=True)

    def _load_scenarios(self) -> List[Scenario]:
        all_train = get_train_scenarios()
        if self.config.scenarios:
            return [s for s in all_train if s.scenario_id in self.config.scenarios]
        return all_train

    def train(self) -> TrainingRun:
        """Run the full training loop. Returns a TrainingRun with all logs."""
        logger.info(f"Starting training run {self._run_id}")
        logger.info(f"Config: {self.config.to_dict()}")
        logger.info(f"Training scenarios: {[s.scenario_id for s in self._train_scenarios]}")

        self.env.start_server()
        logger.info("Flask target server started")

        run = TrainingRun(run_id=self._run_id, config=self.config.to_dict())

        pre_training_scores = self._baseline_evaluation()
        logger.info(f"Pre-training baseline scores: {pre_training_scores}")

        for ep_num in range(1, self.config.num_episodes + 1):
            scenario = self._sample_scenario()
            strategy_id = self.bandit.select(scenario.scenario_id)

            self.agent.reset(strategy_id=strategy_id)

            metrics, trajectory = self.agent.run_episode(
                env=self.env,
                scenario=scenario,
                verbose=self.config.verbose,
            )

            score = score_trajectory(metrics, trajectory, scenario)
            self.bandit.update(scenario.scenario_id, strategy_id, score)
            self.buffer.add(
                scenario_id=scenario.scenario_id,
                strategy_id=strategy_id,
                metrics=metrics,
                trajectory=trajectory,
                scenario=scenario,
            )

            run.log_episode(
                episode_num=ep_num,
                scenario_id=scenario.scenario_id,
                strategy_id=strategy_id,
                metrics=metrics,
                score=score,
            )

            if ep_num % self.config.checkpoint_every == 0:
                self._checkpoint(run, ep_num)
                run.snapshot_bandit(self.bandit, ep_num)

            if self.config.verbose or ep_num % 5 == 0:
                convergence = self.bandit.convergence_summary(scenario.scenario_id)
                logger.info(
                    f"Ep {ep_num:03d} | scenario={scenario.scenario_id} "
                    f"| strategy={strategy_id} | score={score:.4f} "
                    f"| DCS={metrics.get('dcs', 0):.3f} "
                    f"| best_arm={convergence['best_arm']} "
                    f"({convergence['best_arm_pull_fraction']:.1%})"
                )

        post_training_scores = self._post_training_evaluation()
        run.episode_logs.append({
            "type": "summary",
            "pre_training": pre_training_scores,
            "post_training": post_training_scores,
            "improvement": self._compute_improvement(pre_training_scores, post_training_scores),
        })

        run.end_time = time.time()
        path = run.save(self.config.save_dir)
        self.bandit.save(os.path.join(self.config.save_dir, "bandit_final.json"))
        logger.info(f"Training complete. Run saved to {path}")

        return run

    def _sample_scenario(self) -> Scenario:
        """Sample a training scenario with uniform probability."""
        idx = self._rng.integers(0, len(self._train_scenarios))
        return self._train_scenarios[idx]

    def _baseline_evaluation(self) -> Dict[str, float]:
        """
        Run one episode per train scenario using the DEFAULT strategy (no RL).
        Returns mean scores per scenario — establishes the pre-training baseline.
        """
        from src.agent.prompts import DEFAULT_STRATEGY
        scores = {}
        for scenario in self._train_scenarios:
            self.agent.reset(strategy_id=DEFAULT_STRATEGY)
            metrics, traj = self.agent.run_episode(self.env, scenario, verbose=False)
            score = score_trajectory(metrics, traj, scenario)
            scores[scenario.scenario_id] = round(score, 4)
        return scores

    def _post_training_evaluation(self) -> Dict[str, float]:
        """
        Run one episode per train scenario using the BEST bandit arm.
        Returns scores for comparison with pre-training baseline.
        """
        scores = {}
        for scenario in self._train_scenarios:
            best_arm, _ = self.bandit.best_arm(scenario.scenario_id)
            self.agent.reset(strategy_id=best_arm)
            metrics, traj = self.agent.run_episode(self.env, scenario, verbose=False)
            score = score_trajectory(metrics, traj, scenario)
            scores[scenario.scenario_id] = round(score, 4)
        return scores

    def _compute_improvement(
        self, pre: Dict[str, float], post: Dict[str, float]
    ) -> Dict[str, Any]:
        """Compute improvement statistics from pre to post training."""
        improvements = {}
        for sc_id in pre:
            if sc_id in post:
                delta = post[sc_id] - pre[sc_id]
                improvements[sc_id] = {
                    "pre": pre[sc_id],
                    "post": post[sc_id],
                    "delta": round(delta, 4),
                    "relative_improvement_pct": round(delta / max(pre[sc_id], 0.01) * 100, 1),
                }

        pre_vals = list(pre.values())
        post_vals = [post[k] for k in pre if k in post]
        mean_improvement = (
            sum(post_vals) / len(post_vals) - sum(pre_vals) / len(pre_vals)
            if post_vals else 0.0
        )

        return {
            "per_scenario": improvements,
            "mean_score_pre": round(sum(pre_vals) / max(len(pre_vals), 1), 4),
            "mean_score_post": round(sum(post_vals) / max(len(post_vals), 1), 4),
            "mean_improvement": round(mean_improvement, 4),
        }

    def _checkpoint(self, run: TrainingRun, episode_num: int):
        ckpt_dir = os.path.join(self.config.save_dir, f"checkpoint_ep{episode_num:04d}")
        os.makedirs(ckpt_dir, exist_ok=True)
        self.bandit.save(os.path.join(ckpt_dir, "bandit.json"))
        run.save(ckpt_dir)
        logger.debug(f"Checkpoint saved at episode {episode_num}")
