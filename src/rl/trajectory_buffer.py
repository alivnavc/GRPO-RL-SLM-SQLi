"""
Trajectory buffer for storing and ranking agent episodes.

Used by the bandit policy to update arm estimates and by the
DPO-style preference learner to build preference pairs.
"""

import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple
from collections import deque

from src.reward.reward_function import score_trajectory


@dataclass
class Episode:
    """One complete agent episode."""
    episode_id: str
    scenario_id: str
    strategy_id: str
    metrics: Dict[str, Any]
    trajectory: List[Dict[str, Any]]
    score: float
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "Episode":
        return cls(**d)


@dataclass
class PreferencePair:
    """
    A ranked pair of trajectories on the same scenario.
    Used for DPO-style preference learning signal.

    chosen > rejected according to the trajectory score.
    """
    scenario_id: str
    chosen: Episode
    rejected: Episode
    score_gap: float

    @property
    def is_meaningful(self) -> bool:
        """Only meaningful if the gap is large enough to be informative."""
        return self.score_gap >= 0.1


class TrajectoryBuffer:
    """
    Stores episodes per (scenario_id, strategy_id) and provides:
    1. Per-strategy score statistics (for bandit updates)
    2. Preference pairs (for DPO-style learning)
    3. Persistent serialization to disk
    """

    def __init__(self, max_per_strategy: int = 50, save_dir: Optional[str] = None):
        self.max_per_strategy = max_per_strategy
        self.save_dir = save_dir

        self._episodes: Dict[str, deque] = {}
        self._episode_counter = 0

    def add(
        self,
        scenario_id: str,
        strategy_id: str,
        metrics: Dict[str, Any],
        trajectory: List[Dict],
        scenario,
    ) -> Episode:
        """Add a new episode to the buffer. Returns the created Episode."""
        self._episode_counter += 1
        episode_id = f"ep_{self._episode_counter:06d}"
        score = score_trajectory(metrics, trajectory, scenario)

        ep = Episode(
            episode_id=episode_id,
            scenario_id=scenario_id,
            strategy_id=strategy_id,
            metrics=metrics,
            trajectory=trajectory,
            score=score,
        )

        key = f"{scenario_id}::{strategy_id}"
        if key not in self._episodes:
            self._episodes[key] = deque(maxlen=self.max_per_strategy)
        self._episodes[key].append(ep)

        if self.save_dir:
            self._persist_episode(ep)

        return ep

    def get_strategy_stats(self, scenario_id: str, strategy_id: str) -> Dict[str, float]:
        """Return mean, std, count of scores for a strategy on a scenario."""
        key = f"{scenario_id}::{strategy_id}"
        episodes = list(self._episodes.get(key, []))
        if not episodes:
            return {"mean": 0.0, "std": 0.0, "count": 0, "best": 0.0}
        scores = [ep.score for ep in episodes]
        n = len(scores)
        mean = sum(scores) / n
        variance = sum((s - mean) ** 2 for s in scores) / max(n - 1, 1)
        std = variance ** 0.5
        return {
            "mean": round(mean, 4),
            "std": round(std, 4),
            "count": n,
            "best": round(max(scores), 4),
        }

    def get_all_strategy_stats(self, scenario_id: str) -> Dict[str, Dict]:
        """Return stats for all strategies seen on a scenario."""
        stats = {}
        for key in self._episodes:
            sc_id, strat_id = key.split("::", 1)
            if sc_id == scenario_id:
                stats[strat_id] = self.get_strategy_stats(scenario_id, strat_id)
        return stats

    def build_preference_pairs(
        self, scenario_id: str, min_gap: float = 0.05
    ) -> List[PreferencePair]:
        """
        Build DPO-style preference pairs for a scenario.

        For each strategy, find the best episode. Then pair strategies
        where one clearly dominates the other (score_gap >= min_gap).
        """
        best_by_strategy: Dict[str, Episode] = {}
        for key, episodes in self._episodes.items():
            sc_id, strat_id = key.split("::", 1)
            if sc_id != scenario_id:
                continue
            best = max(episodes, key=lambda e: e.score)
            best_by_strategy[strat_id] = best

        pairs = []
        strategy_ids = list(best_by_strategy.keys())
        for i in range(len(strategy_ids)):
            for j in range(i + 1, len(strategy_ids)):
                ep_a = best_by_strategy[strategy_ids[i]]
                ep_b = best_by_strategy[strategy_ids[j]]
                if ep_a.score > ep_b.score:
                    chosen, rejected = ep_a, ep_b
                else:
                    chosen, rejected = ep_b, ep_a
                gap = abs(ep_a.score - ep_b.score)
                if gap >= min_gap:
                    pairs.append(PreferencePair(
                        scenario_id=scenario_id,
                        chosen=chosen,
                        rejected=rejected,
                        score_gap=round(gap, 4),
                    ))
        return sorted(pairs, key=lambda p: p.score_gap, reverse=True)

    def total_episodes(self) -> int:
        return sum(len(q) for q in self._episodes.values())

    def _persist_episode(self, episode: Episode):
        """Save episode to disk as JSONL."""
        os.makedirs(self.save_dir, exist_ok=True)
        path = os.path.join(self.save_dir, f"{episode.scenario_id}_episodes.jsonl")
        with open(path, "a") as f:
            f.write(json.dumps(episode.to_dict()) + "\n")

    def load_from_dir(self, save_dir: str):
        """Reload episodes from JSONL files in save_dir."""
        if not os.path.isdir(save_dir):
            return
        for fname in os.listdir(save_dir):
            if not fname.endswith("_episodes.jsonl"):
                continue
            fpath = os.path.join(save_dir, fname)
            with open(fpath) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                        ep = Episode.from_dict(d)
                        key = f"{ep.scenario_id}::{ep.strategy_id}"
                        if key not in self._episodes:
                            self._episodes[key] = deque(maxlen=self.max_per_strategy)
                        self._episodes[key].append(ep)
                        self._episode_counter += 1
                    except Exception:
                        pass
