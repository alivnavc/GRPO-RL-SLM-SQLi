"""
GRPO (Group Relative Policy Optimization) training loop for the SQLi agent.

Algorithm (DeepSeek-R1 style, adapted for multi-step tool-calling episodes):

  For each training step:
    1. Sample a scenario from the training set
    2. Run G complete episodes with the current policy → collect per-episode:
         - All (system_prompt, user_prompt, completion_ids) tuples
         - Total episode reward
    3. Compute advantages:
         advantage_i = (reward_i - mean(rewards)) / (std(rewards) + eps)
    4. For each episode i, recompute log_probs via teacher-forcing (with grad)
       and ref_log_probs (no grad), then compute GRPO loss:
         L_i = -advantage_i * mean_tokens(log_p_theta - beta * (log_p_theta - log_p_ref))
         L   = mean_G(L_i)
    5. Backprop + AdamW step + zero_grad

Why episode-level (not step-level) GRPO:
  - Pen testing success is episodic: DCS only meaningful at episode end.
  - Step-level rewards are dense (injection signals) but the terminal DCS bonus
    dominates. Episode-level GRPO aligns the credit assignment with the actual
    objective — find all vulns efficiently.
"""

import os
import gc
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.agent.prompts import build_system_prompt, build_task_prompt, DEFAULT_STRATEGY
from src.agent.tools import parse_llm_output
from src.environment.sqli_env import SQLiEnvironment
from src.environment.scenarios import get_train_scenarios

from grpo_pipeline.agent.local_llm import LocalLLM
from grpo_pipeline.agent.grpo_prompts import build_grpo_system_prompt


# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------

@dataclass
class GRPOConfig:
    G: int = 4                          # rollouts per scenario per step
    beta: float = 0.01                  # KL penalty coefficient
    lr: float = 2e-5                    # AdamW learning rate
    max_steps_per_episode: int = 20     # hard step cap per episode
    strategy: str = DEFAULT_STRATEGY   # system prompt strategy
    checkpoint_dir: str = "results/grpo_checkpoints"
    grad_clip: float = 1.0


# ------------------------------------------------------------------
# Rollout collection
# ------------------------------------------------------------------

@dataclass
class StepRecord:
    system_prompt: str
    user_prompt: str
    completion_ids: List[int]
    reward: float


@dataclass
class EpisodeRollout:
    steps: List[StepRecord] = field(default_factory=list)
    total_reward: float = 0.0
    num_findings: int = 0
    termination: str = "budget"


def _run_episode(
    llm: LocalLLM,
    env: SQLiEnvironment,
    scenario: Any,
    config: GRPOConfig,
) -> EpisodeRollout:
    """
    Run one full episode collecting (prompt, completion_ids, step_reward) per step.
    No gradients — generation is inference-only. Gradients are recomputed later.
    """
    state = env.reset(scenario)
    system_prompt = build_grpo_system_prompt(build_system_prompt(config.strategy))
    rollout = EpisodeRollout()

    for step_idx in range(config.max_steps_per_episode):
        user_prompt = build_task_prompt(
            state_context=state.to_prompt_context(),
            scenario_description=scenario.description,
        )

        completion_text, completion_ids = llm.generate(system_prompt, user_prompt)
        action = parse_llm_output(completion_text)

        if not action.is_valid:
            action.tool = "stop"
            action.params = {"reason": "parse_error"}

        # Minimum-steps guard: force enumerate on step 0, prevent premature stop
        if step_idx == 0 and action.tool != "enumerate_endpoints":
            action.tool = "enumerate_endpoints"
            action.params = {}
        elif step_idx < 2 and action.tool == "stop":
            action.tool = "enumerate_endpoints"
            action.params = {}

        next_state, reward, done, _info = env.step(action.to_env_action())
        rollout.steps.append(
            StepRecord(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                completion_ids=completion_ids,
                reward=reward,
            )
        )
        rollout.total_reward += reward
        state = next_state

        if done:
            rollout.termination = "agent_stopped" if action.tool == "stop" else "budget"
            break

    rollout.num_findings = len(state.findings)
    return rollout


# ------------------------------------------------------------------
# GRPO loss
# ------------------------------------------------------------------

def _compute_grpo_loss(
    llm: LocalLLM,
    rollouts: List[EpisodeRollout],
    config: GRPOConfig,
) -> torch.Tensor:
    """
    Compute the GRPO loss over G episode rollouts.

    Loss per episode:
        L_i = -advantage_i * mean_tokens(log_p_theta(t) - beta*(log_p_theta(t) - log_p_ref(t)))

    Aggregated:
        L = mean_G(L_i)

    The KL term (beta * (log_p_theta - log_p_ref)) prevents the policy from
    drifting too far from the reference model (base Qwen weights).
    """
    rewards = [r.total_reward for r in rollouts]
    mean_r = sum(rewards) / len(rewards)
    std_r = (sum((r - mean_r) ** 2 for r in rewards) / len(rewards)) ** 0.5
    eps = 1e-8

    episode_losses: List[torch.Tensor] = []

    for rollout, reward in zip(rollouts, rewards):
        advantage = (reward - mean_r) / (std_r + eps)

        if len(rollout.steps) == 0:
            continue

        step_losses: List[torch.Tensor] = []
        for step in rollout.steps:
            if len(step.completion_ids) == 0:
                continue

            # Policy log probs (with gradient)
            policy_lp = llm.compute_log_probs(
                step.system_prompt,
                step.user_prompt,
                step.completion_ids,
                use_ref=False,
            )

            # Reference log probs (no gradient)
            ref_lp = llm.compute_log_probs(
                step.system_prompt,
                step.user_prompt,
                step.completion_ids,
                use_ref=True,
            )

            # Per-token objective: log_p_theta - beta * KL
            per_token = policy_lp - config.beta * (policy_lp - ref_lp.detach())
            step_losses.append(per_token.mean())

        if not step_losses:
            continue

        episode_obj = torch.stack(step_losses).mean()
        episode_losses.append(-advantage * episode_obj)

    if not episode_losses:
        return torch.tensor(0.0, requires_grad=True)

    return torch.stack(episode_losses).mean()


# ------------------------------------------------------------------
# Trainer
# ------------------------------------------------------------------

class GRPOTrainer:
    """
    Orchestrates GRPO training:
      - collect G rollouts per scenario
      - compute advantages + GRPO loss
      - backprop LoRA weights
      - save checkpoint per epoch
    """

    def __init__(self, llm: LocalLLM, env: SQLiEnvironment, config: GRPOConfig):
        self.llm = llm
        self.env = env
        self.config = config
        self.optimizer = torch.optim.AdamW(
            [p for p in llm.model.parameters() if p.requires_grad],
            lr=config.lr,
        )
        self.train_log: List[Dict] = []

    def train_epoch(self, epoch: int) -> Dict:
        """Run one epoch: one GRPO step per training scenario."""
        scenarios = get_train_scenarios()
        epoch_losses = []
        epoch_rewards = []

        for scenario in scenarios:
            rollouts = [
                _run_episode(self.llm, self.env, scenario, self.config)
                for _ in range(self.config.G)
            ]
            gc.collect()

            rewards = [r.total_reward for r in rollouts]
            epoch_rewards.extend(rewards)

            # Skip update if all rollouts identical (std=0) — no learning signal
            if max(rewards) - min(rewards) < 1e-6:
                continue

            self.optimizer.zero_grad()
            loss = _compute_grpo_loss(self.llm, rollouts, self.config)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.llm.model.parameters() if p.requires_grad],
                self.config.grad_clip,
            )
            self.optimizer.step()
            epoch_losses.append(loss.item())
            del loss
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        result = {
            "epoch": epoch,
            "mean_loss": sum(epoch_losses) / max(len(epoch_losses), 1),
            "mean_reward": sum(epoch_rewards) / max(len(epoch_rewards), 1),
            "min_reward": min(epoch_rewards) if epoch_rewards else 0.0,
            "max_reward": max(epoch_rewards) if epoch_rewards else 0.0,
        }
        self.train_log.append(result)

        ckpt_path = os.path.join(self.config.checkpoint_dir, f"epoch_{epoch:03d}")
        self.llm.save_lora(ckpt_path)

        return result
