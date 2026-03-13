"""
Reward function translating Part-1 KPIs into a scalar reward signal.

Design decisions documented inline. This is the most critical component —
poor reward design is the most common failure mode in RL for LLM agents.
"""

from typing import Any, Dict, Optional
from src.environment.scenarios import Scenario


# ---------------------------------------------------------------------------
# Reward weights (easily tunable)
# ---------------------------------------------------------------------------

REWARD_CONFIG = {
    # --- Terminal rewards (end-of-episode) ---
    "r_new_true_positive": 5.0,     # Confirmed new injection point found
    "r_confirmation_bonus": 2.0,    # Finding matches known ground-truth vuln
    "r_false_positive_penalty": -3.0,  # Reported finding that is NOT a real vuln
    "r_clean_episode_bonus": 3.0,   # Correctly found 0 vulns on a clean target

    # --- Shaped step rewards (intermediate signals) ---
    "r_sql_error_signal": 1.5,      # Response contains SQL error → productive
    "r_time_delay_signal": 1.0,     # Response time > 2s → possible time-based blind
    "r_credential_exposure": 2.0,   # Response exposes credentials/schema → high value
    "r_new_endpoint_found": 0.5,    # Agent discovered a new endpoint
    "r_new_param_tested": 0.3,      # Agent tested a parameter it hadn't tested before

    # --- Efficiency penalties ---
    "r_step_cost": -0.1,            # Per-step cost (encourages efficiency)
    "r_duplicate_penalty": -0.5,    # Same param+payload tested again
    "r_dead_end_penalty": -0.2,     # Step produced zero new information (added on top of step_cost)

    # --- Autonomy bonuses ---
    "r_dead_end_recovery": 0.8,     # Agent changed strategy after 3+ dead ends
    "r_clean_stop": 1.0,            # Agent stopped before budget when DCS≥0.8
    "r_over_run_penalty": -1.0,     # Agent continued 10+ steps after full coverage

    # --- Reasoning quality (computed periodically) ---
    "r_attack_diversity_bonus": 1.0,  # Agent used ≥3 distinct attack types in episode
}


# ---------------------------------------------------------------------------
# Step-level reward
# ---------------------------------------------------------------------------

def compute_step_reward(
    state,           # AgentState
    action: Dict[str, Any],
    result: Dict[str, Any],
    scenario: Scenario,
    productive: bool,
) -> float:
    """
    Compute the immediate reward for a single environment step.

    Key design choices:
    1. SPARSE + SHAPED: The main reward is sparse (only on confirmed finding),
       but we add intermediate shaped signals for SQL error detection and
       new endpoint discovery to reduce the credit assignment problem.

    2. FALSE POSITIVE PENALTY > STEP REWARD: Reporting a false positive
       costs -3.0, which far exceeds the cumulative step rewards an agent
       could gain by randomly reporting things. This prevents reward hacking.

    3. STEP COST: A small -0.1 per step forces the agent to be efficient.
       Without this, agents tend to enumerate endlessly.

    4. NO REWARD FOR THOUGHT: We only reward actions with observable effects.
       analyze_response with no payload gets 0 shaped reward — agents can't
       farm rewards by "thinking" without acting.
    """
    reward = 0.0
    tool = action.get("tool", "")
    params = action.get("params", {})

    # --- Per-step cost (always applied) ---
    reward += REWARD_CONFIG["r_step_cost"]

    # --- Dead-end penalty (no new info) ---
    if not productive and tool not in ("stop", "analyze_response"):
        reward += REWARD_CONFIG["r_dead_end_penalty"]

    # --- Tool-specific rewards ---

    if tool == "enumerate_endpoints":
        newly_discovered = result.get("discovered", [])
        if newly_discovered:
            reward += REWARD_CONFIG["r_new_endpoint_found"] * len(newly_discovered)

    elif tool == "inject_payload":
        signals = result.get("injection_signals", [])
        reward += _score_injection_signals(signals)
        if "error" not in result:
            reward += REWARD_CONFIG["r_new_param_tested"]

    elif tool == "report_finding":
        if result.get("recorded", False):
            reward += _score_finding(state, params, scenario)
        else:
            reward += REWARD_CONFIG["r_duplicate_penalty"]

    elif tool == "stop":
        reward += _score_stop(state, scenario)

    return round(reward, 4)


def _score_injection_signals(signals: list) -> float:
    """Reward based on what injection signals were detected in the response."""
    reward = 0.0
    for signal in signals:
        if signal.startswith("sql_error"):
            reward += REWARD_CONFIG["r_sql_error_signal"]
        elif signal.startswith("time_delay"):
            reward += REWARD_CONFIG["r_time_delay_signal"]
        elif signal == "credential_exposure" or signal == "schema_exposure":
            reward += REWARD_CONFIG["r_credential_exposure"]
    return reward


def _score_finding(state, params: Dict, scenario: Scenario) -> float:
    """
    Reward for report_finding tool call.

    True positive (matches ground truth) → large reward
    False positive (not in ground truth) → penalty

    We verify against ground truth here rather than waiting for
    episode end to reduce reward sparsity.
    """
    endpoint = params.get("endpoint", "")
    param = params.get("param", "")
    confidence = float(params.get("confidence", 0.5))

    is_true_positive = _check_ground_truth(endpoint, param, scenario)

    if is_true_positive:
        base = REWARD_CONFIG["r_new_true_positive"]
        confirmation_bonus = REWARD_CONFIG["r_confirmation_bonus"]
        confidence_scale = 0.5 + 0.5 * confidence
        return (base + confirmation_bonus) * confidence_scale
    else:
        return REWARD_CONFIG["r_false_positive_penalty"] * (0.5 + 0.5 * confidence)


def _check_ground_truth(endpoint: str, param: str, scenario: Scenario) -> bool:
    """Check if endpoint+param matches any known vulnerability in the scenario."""
    for vuln in scenario.active_vulns:
        vuln_ep = vuln.endpoint.replace("{id}", "").rstrip("/")
        if (vuln_ep in endpoint or endpoint in vuln_ep) and vuln.param == param:
            return True
    return False


def _score_stop(state, scenario: Scenario) -> float:
    """
    Reward for explicit stop action.

    Design: We reward stopping ONLY if it was the right call.
    - Clean stop with high DCS: bonus
    - Stopping early with low DCS and vulns remaining: no bonus (rely on episode-end scoring)
    - Stopping on a clean target correctly: bonus
    """
    known_count = len(scenario.active_vulns)
    findings_count = len(state.findings)

    if known_count == 0 and findings_count == 0:
        return REWARD_CONFIG["r_clean_episode_bonus"]

    dcs_approx = min(findings_count / max(known_count, 1), 1.0)

    if dcs_approx >= 0.8 and state.step < state.step_budget * 0.8:
        return REWARD_CONFIG["r_clean_stop"]

    return 0.0


# ---------------------------------------------------------------------------
# Episode-level terminal reward (called once at end of episode)
# ---------------------------------------------------------------------------

def compute_terminal_reward(metrics: Dict[str, Any], scenario: Scenario) -> float:
    """
    Bonus/penalty applied at episode end based on final KPI values.
    This captures things that can only be assessed after the full episode.

    Not added to per-step rewards — used separately in trajectory scoring
    for the bandit policy update.
    """
    reward = 0.0

    dcs = metrics.get("dcs", 0.0)
    stc = metrics.get("stc_score", 0.0)
    avds = metrics.get("avds", 0.0)
    fp_count = metrics.get("false_positives", 0)
    known_count = len(scenario.active_vulns)

    reward += 10.0 * dcs

    reward += 3.0 * stc

    if avds >= 0.5:
        reward += REWARD_CONFIG["r_attack_diversity_bonus"]

    reward -= 2.0 * fp_count

    if dcs >= 0.9 and fp_count == 0 and known_count > 0:
        reward += 5.0

    if metrics.get("termination_reason") == "budget_exhausted" and dcs < 0.5:
        reward -= 2.0

    return round(reward, 4)


# ---------------------------------------------------------------------------
# Trajectory scoring (used by bandit policy update)
# ---------------------------------------------------------------------------

def score_trajectory(
    metrics: Dict[str, Any],
    trajectory: list,
    scenario: Scenario,
) -> float:
    """
    Compute a single scalar score for a complete trajectory.
    Used by the bandit to update arm estimates.

    Formula:
        score = 0.4*DCS + 0.25*STC + 0.15*AVDS + 0.1*DERR + 0.1*TIE
              + terminal_bonus
              - 0.5 * FP_rate

    This directly implements the composite_task_score from kpi_framework.md
    with DCS weighted highest.
    """
    dcs = metrics.get("dcs", 0.0)
    stc = metrics.get("stc_score", 0.0)
    avds = metrics.get("avds", 0.0)
    derr = metrics.get("derr", 1.0)
    tie = metrics.get("tie", 0.0)
    fp_rate = metrics.get("false_positives", 0) / max(metrics.get("findings_count", 1), 1)

    base_score = (
        0.40 * dcs
        + 0.25 * stc
        + 0.15 * avds
        + 0.10 * derr
        + 0.10 * tie
        - 0.50 * fp_rate
    )

    terminal = compute_terminal_reward(metrics, scenario)
    normalized_terminal = terminal / 20.0

    return round(base_score + normalized_terminal, 4)
