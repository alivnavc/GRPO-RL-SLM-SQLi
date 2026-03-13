"""
Tests for the reward function and trajectory scoring.
"""

import pytest
from src.environment.scenarios import get_scenario
from src.reward.reward_function import (
    REWARD_CONFIG,
    compute_step_reward,
    compute_terminal_reward,
    score_trajectory,
    _check_ground_truth,
    _score_injection_signals,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def easy_scenario():
    return get_scenario("train_easy_single")


@pytest.fixture
def full_scenario():
    return get_scenario("train_full")


@pytest.fixture
def clean_scenario():
    return get_scenario("holdout_clean")


class MockState:
    def __init__(self, step=1, step_budget=20, findings=None):
        self.step = step
        self.step_budget = step_budget
        self.findings = findings or []
        self.tokens_used = 0
        self.token_budget = 8000
        self.episode_done = False
        self.termination_reason = ""


# ---------------------------------------------------------------------------
# Injection signal scoring
# ---------------------------------------------------------------------------

class TestInjectionSignalScoring:

    def test_sql_error_positive_reward(self):
        signals = ["sql_error:sqlite"]
        reward = _score_injection_signals(signals)
        assert reward == REWARD_CONFIG["r_sql_error_signal"]

    def test_time_delay_positive_reward(self):
        signals = ["time_delay:>2s"]
        reward = _score_injection_signals(signals)
        assert reward == REWARD_CONFIG["r_time_delay_signal"]

    def test_credential_exposure_high_reward(self):
        signals = ["credential_exposure"]
        reward = _score_injection_signals(signals)
        assert reward == REWARD_CONFIG["r_credential_exposure"]

    def test_multiple_signals_additive(self):
        signals = ["sql_error:sqlite", "credential_exposure"]
        reward = _score_injection_signals(signals)
        expected = REWARD_CONFIG["r_sql_error_signal"] + REWARD_CONFIG["r_credential_exposure"]
        assert reward == pytest.approx(expected)

    def test_empty_signals_zero_reward(self):
        assert _score_injection_signals([]) == 0.0


# ---------------------------------------------------------------------------
# Ground truth checking
# ---------------------------------------------------------------------------

class TestGroundTruthCheck:

    def test_true_positive_login(self, easy_scenario):
        assert _check_ground_truth("/api/login", "username", easy_scenario) is True

    def test_false_positive_health(self, easy_scenario):
        assert _check_ground_truth("/api/health", "status", easy_scenario) is False

    def test_false_positive_safe_endpoint(self, easy_scenario):
        assert _check_ground_truth("/api/products", "q", easy_scenario) is False

    def test_true_positive_full_scenario(self, full_scenario):
        assert _check_ground_truth("/api/products/search", "q", full_scenario) is True
        assert _check_ground_truth("/api/orders", "status", full_scenario) is True

    def test_clean_scenario_always_false(self, clean_scenario):
        assert _check_ground_truth("/api/login", "username", clean_scenario) is False


# ---------------------------------------------------------------------------
# Step reward
# ---------------------------------------------------------------------------

class TestStepReward:

    def test_step_cost_always_applied(self, easy_scenario):
        state = MockState()
        action = {"tool": "http_request", "params": {}}
        result = {"status_code": 200, "body": "ok"}
        reward = compute_step_reward(state, action, result, easy_scenario, productive=False)
        assert reward <= REWARD_CONFIG["r_step_cost"]

    def test_new_endpoint_positive(self, easy_scenario):
        state = MockState()
        action = {"tool": "enumerate_endpoints", "params": {}}
        result = {"discovered": ["/api/login", "/api/products"]}
        reward = compute_step_reward(state, action, result, easy_scenario, productive=True)
        assert reward > 0

    def test_true_positive_report_positive(self, easy_scenario):
        state = MockState()
        action = {
            "tool": "report_finding",
            "params": {
                "endpoint": "/api/login",
                "param": "username",
                "payload": "' OR '1'='1",
                "vuln_type": "error_based",
                "confidence": 0.9,
                "evidence": "SQL error",
            },
        }
        result = {"recorded": True}
        reward = compute_step_reward(state, action, result, easy_scenario, productive=True)
        assert reward > 0

    def test_false_positive_report_negative(self, easy_scenario):
        state = MockState()
        action = {
            "tool": "report_finding",
            "params": {
                "endpoint": "/api/health",
                "param": "status",
                "payload": "' OR 1=1",
                "vuln_type": "error_based",
                "confidence": 0.9,
                "evidence": "imagined",
            },
        }
        result = {"recorded": True}
        reward = compute_step_reward(state, action, result, easy_scenario, productive=True)
        assert reward < 0

    def test_duplicate_report_penalized(self, easy_scenario):
        state = MockState()
        action = {"tool": "report_finding", "params": {"endpoint": "/api/login", "param": "username"}}
        result = {"recorded": False}
        reward = compute_step_reward(state, action, result, easy_scenario, productive=False)
        assert reward < 0

    def test_inject_with_signals_positive(self, easy_scenario):
        state = MockState()
        action = {
            "tool": "inject_payload",
            "params": {"endpoint": "/api/login", "method": "POST", "param": "username",
                       "payload": "'", "attack_type": "error_based"},
        }
        result = {"injection_signals": ["sql_error:sqlite"], "status_code": 500}
        reward = compute_step_reward(state, action, result, easy_scenario, productive=True)
        assert reward > 0

    def test_dead_end_extra_penalty(self, easy_scenario):
        state = MockState()
        action = {"tool": "http_request", "params": {"url": "/api/health"}}
        result = {"status_code": 200, "body": "ok", "injection_signals": []}
        reward_productive = compute_step_reward(state, action, result, easy_scenario, productive=True)
        reward_dead_end = compute_step_reward(state, action, result, easy_scenario, productive=False)
        assert reward_dead_end < reward_productive


# ---------------------------------------------------------------------------
# Terminal reward
# ---------------------------------------------------------------------------

class TestTerminalReward:

    def test_perfect_episode_high_reward(self, easy_scenario):
        metrics = {
            "dcs": 1.0,
            "stc_score": 0.9,
            "avds": 0.75,
            "false_positives": 0,
            "termination_reason": "agent_stopped",
        }
        reward = compute_terminal_reward(metrics, easy_scenario)
        assert reward > 10.0

    def test_zero_dcs_low_reward(self, easy_scenario):
        metrics = {
            "dcs": 0.0,
            "stc_score": 0.0,
            "avds": 0.0,
            "false_positives": 0,
            "termination_reason": "budget_exhausted",
        }
        reward = compute_terminal_reward(metrics, easy_scenario)
        assert reward < 5.0

    def test_false_positives_penalized(self, easy_scenario):
        metrics_clean = {"dcs": 0.5, "stc_score": 0.5, "avds": 0.5, "false_positives": 0, "termination_reason": ""}
        metrics_fp = {"dcs": 0.5, "stc_score": 0.5, "avds": 0.5, "false_positives": 3, "termination_reason": ""}
        r_clean = compute_terminal_reward(metrics_clean, easy_scenario)
        r_fp = compute_terminal_reward(metrics_fp, easy_scenario)
        assert r_fp < r_clean

    def test_budget_exhausted_with_low_dcs_penalty(self, easy_scenario):
        metrics = {
            "dcs": 0.3,
            "stc_score": 0.0,
            "avds": 0.0,
            "false_positives": 0,
            "termination_reason": "budget_exhausted",
        }
        reward = compute_terminal_reward(metrics, easy_scenario)
        assert reward < 5.0


# ---------------------------------------------------------------------------
# Trajectory scoring
# ---------------------------------------------------------------------------

class TestTrajectoryScoring:

    def test_high_dcs_high_score(self, easy_scenario):
        metrics = {
            "dcs": 0.9,
            "stc_score": 0.8,
            "avds": 0.75,
            "derr": 1.0,
            "tie": 0.8,
            "false_positives": 0,
            "findings_count": 1,
        }
        score = score_trajectory(metrics, [], easy_scenario)
        assert score > 0.5

    def test_zero_metrics_low_score(self, easy_scenario):
        metrics = {
            "dcs": 0.0,
            "stc_score": 0.0,
            "avds": 0.0,
            "derr": 0.0,
            "tie": 0.0,
            "false_positives": 0,
            "findings_count": 0,
        }
        score = score_trajectory(metrics, [], easy_scenario)
        assert score <= 0.2

    def test_fp_lowers_score(self, easy_scenario):
        metrics_clean = {
            "dcs": 0.6, "stc_score": 0.6, "avds": 0.5, "derr": 1.0, "tie": 0.7,
            "false_positives": 0, "findings_count": 1,
        }
        metrics_fp = {
            "dcs": 0.6, "stc_score": 0.6, "avds": 0.5, "derr": 1.0, "tie": 0.7,
            "false_positives": 2, "findings_count": 3,
        }
        s_clean = score_trajectory(metrics_clean, [], easy_scenario)
        s_fp = score_trajectory(metrics_fp, [], easy_scenario)
        assert s_fp < s_clean

    def test_scores_bounded_zero_to_one_ish(self, easy_scenario):
        metrics = {
            "dcs": 0.5, "stc_score": 0.5, "avds": 0.5, "derr": 0.5, "tie": 0.5,
            "false_positives": 0, "findings_count": 1,
        }
        score = score_trajectory(metrics, [], easy_scenario)
        assert -1.0 <= score <= 2.0
