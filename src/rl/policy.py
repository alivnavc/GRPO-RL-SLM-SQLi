"""
Multi-Armed Bandit policy over attack strategy variants.

WHY BANDIT OVER PPO:
  Full PPO requires a differentiable policy network over a continuous action
  space. For LLM agents, the "policy" is largely encoded in the prompt
  strategy. The meaningful discrete choice is WHICH STRATEGY VARIANT to use
  for a given scenario type. A bandit over these 5 strategy arms is:
    - Implementable without GPU training
    - Measurably improvable (arm selection shifts toward better strategies)
    - Directly connected to the KPI framework
    - Honest about what we're actually optimizing

  We use UCB1 (Upper Confidence Bound) which balances exploration of
  under-tried strategies with exploitation of known-good ones.

UCB1 formula:
    score(arm) = mean_reward(arm) + C * sqrt(ln(N) / n(arm))
    where:
        C       = exploration constant (default 1.41 ≈ sqrt(2))
        N       = total pulls across all arms
        n(arm)  = pulls for this arm
"""

import math
import json
import os
from typing import Dict, List, Optional, Tuple

from src.agent.prompts import get_all_strategy_ids, STRATEGY_VARIANTS


class UCBBandit:
    """
    Upper Confidence Bound (UCB1) bandit over attack strategy variants.

    Each "arm" is a strategy from prompts.STRATEGY_VARIANTS.
    Rewards come from trajectory scores computed by reward_function.score_trajectory.
    """

    def __init__(
        self,
        strategy_ids: Optional[List[str]] = None,
        exploration_constant: float = 1.41,
        scenario_scoped: bool = True,
    ):
        self.arms = strategy_ids or get_all_strategy_ids()
        self.C = exploration_constant
        self.scenario_scoped = scenario_scoped

        self._counts: Dict[str, Dict[str, int]] = {}
        self._rewards: Dict[str, Dict[str, float]] = {}
        self._total_pulls: Dict[str, int] = {}

    def _ensure_scenario(self, scenario_id: str):
        if scenario_id not in self._counts:
            self._counts[scenario_id] = {arm: 0 for arm in self.arms}
            self._rewards[scenario_id] = {arm: 0.0 for arm in self.arms}
            self._total_pulls[scenario_id] = 0

    def select(self, scenario_id: str) -> str:
        """
        Select the best strategy for a given scenario using UCB1.

        Untried arms are always selected first (infinite UCB score).
        After all arms tried once, UCB1 formula takes over.
        """
        self._ensure_scenario(scenario_id)
        N = self._total_pulls[scenario_id]

        for arm in self.arms:
            if self._counts[scenario_id][arm] == 0:
                return arm

        ucb_scores = {}
        for arm in self.arms:
            n = self._counts[scenario_id][arm]
            mean = self._rewards[scenario_id][arm] / n
            confidence = self.C * math.sqrt(math.log(N) / n)
            ucb_scores[arm] = mean + confidence

        return max(ucb_scores, key=ucb_scores.__getitem__)

    def update(self, scenario_id: str, strategy_id: str, reward: float):
        """Update the bandit with the result of playing an arm."""
        self._ensure_scenario(scenario_id)
        self._counts[scenario_id][strategy_id] += 1
        self._rewards[scenario_id][strategy_id] += reward
        self._total_pulls[scenario_id] += 1

    def get_arm_stats(self, scenario_id: str) -> Dict[str, Dict]:
        """Return current statistics for all arms on a scenario."""
        self._ensure_scenario(scenario_id)
        stats = {}
        N = self._total_pulls[scenario_id]
        for arm in self.arms:
            n = self._counts[scenario_id][arm]
            mean = self._rewards[scenario_id][arm] / max(n, 1)
            ucb = (
                mean + self.C * math.sqrt(math.log(max(N, 1)) / n)
                if n > 0 else float("inf")
            )
            stats[arm] = {
                "pulls": n,
                "mean_reward": round(mean, 4),
                "ucb_score": round(ucb, 4) if ucb != float("inf") else "inf",
                "total_reward": round(self._rewards[scenario_id][arm], 4),
                "description": STRATEGY_VARIANTS[arm]["description"],
            }
        return stats

    def best_arm(self, scenario_id: str) -> Tuple[str, float]:
        """Return (strategy_id, mean_reward) of the empirically best arm."""
        self._ensure_scenario(scenario_id)
        best = max(
            self.arms,
            key=lambda a: self._rewards[scenario_id][a] / max(self._counts[scenario_id][a], 1),
        )
        mean = self._rewards[scenario_id][best] / max(self._counts[scenario_id][best], 1)
        return best, round(mean, 4)

    def convergence_summary(self, scenario_id: str) -> Dict:
        """
        Summarize whether the bandit has converged.
        Convergence criterion: best arm selected ≥60% of last N pulls.
        """
        self._ensure_scenario(scenario_id)
        total = self._total_pulls[scenario_id]
        best, best_mean = self.best_arm(scenario_id)
        best_pulls = self._counts[scenario_id][best]
        concentration = best_pulls / max(total, 1)

        return {
            "total_pulls": total,
            "best_arm": best,
            "best_mean_reward": best_mean,
            "best_arm_pull_fraction": round(concentration, 3),
            "converged": concentration >= 0.6 and total >= len(self.arms) * 2,
        }

    def save(self, path: str):
        """Persist bandit state to JSON."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {
            "arms": self.arms,
            "exploration_constant": self.C,
            "counts": self._counts,
            "rewards": self._rewards,
            "total_pulls": self._total_pulls,
        }
        with open(path, "w") as f:
            json.dump(state, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "UCBBandit":
        """Load bandit from JSON checkpoint."""
        with open(path) as f:
            state = json.load(f)
        bandit = cls(
            strategy_ids=state["arms"],
            exploration_constant=state["exploration_constant"],
        )
        bandit._counts = state["counts"]
        bandit._rewards = state["rewards"]
        bandit._total_pulls = state["total_pulls"]
        return bandit
