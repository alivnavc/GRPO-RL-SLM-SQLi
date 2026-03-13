"""
Tests for the vulnerable Flask app and RL environment wrapper.
"""

import pytest
import threading
import time
import requests

from src.environment.vulnerable_app import create_app, VULNERABILITY_CATALOGUE
from src.environment.sqli_env import SQLiEnvironment
from src.environment.scenarios import get_scenario, SCENARIOS


# ---------------------------------------------------------------------------
# Flask app fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def flask_app():
    app = create_app(db_path=":memory:")
    app.config["TESTING"] = True
    return app


@pytest.fixture(scope="module")
def client(flask_app):
    return flask_app.test_client()


# ---------------------------------------------------------------------------
# Vulnerable app tests
# ---------------------------------------------------------------------------

class TestVulnerableApp:

    def test_health_endpoint(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "healthy"

    def test_login_valid_credentials(self, client):
        resp = client.post("/api/login", json={"username": "alice", "password": "alice123"})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "ok"

    def test_login_invalid_credentials(self, client):
        resp = client.post("/api/login", json={"username": "alice", "password": "wrongpass"})
        assert resp.status_code == 401

    def test_login_sqli_error_based(self, client):
        """VULN-1: Single quote triggers SQL error."""
        resp = client.post("/api/login", json={"username": "'", "password": "x"})
        assert resp.status_code == 500
        data = resp.get_json()
        assert "error" in data["status"]

    def test_login_sqli_bypass(self, client):
        """VULN-1: Comment-based bypass drops the AND password= clause."""
        payload = "' OR '1'='1'--"
        resp = client.post("/api/login", json={"username": payload, "password": "x"})
        assert resp.status_code == 200

    def test_search_normal(self, client):
        resp = client.get("/api/products/search?q=Laptop")
        assert resp.status_code == 200
        data = resp.get_json()
        assert isinstance(data["results"], list)

    def test_search_sqli_union(self, client):
        """VULN-2: UNION injection exposes schema."""
        payload = "' UNION SELECT name,sql,null FROM sqlite_master--"
        resp = client.get(f"/api/products/search?q={payload}")
        assert resp.status_code in (200, 500)

    def test_product_detail_sqli(self, client):
        """VULN-3: Path param injectable."""
        resp = client.get("/api/products/1'")
        assert resp.status_code == 500

    def test_safe_products_list(self, client):
        """Safe parameterized endpoint — must NOT produce errors."""
        resp = client.get("/api/products")
        assert resp.status_code == 200

    def test_orders_blind_sqli(self, client):
        """VULN-4: Blind injection — error swallowed but count differs."""
        resp_true = client.get("/api/orders?status=pending' AND '1'='1")
        resp_false = client.get("/api/orders?status=pending' AND '1'='2")
        assert resp_true.status_code == 200
        assert resp_false.status_code == 200
        data_true = resp_true.get_json()
        data_false = resp_false.get_json()
        assert data_true["count"] != data_false["count"]

    def test_vulnerability_catalogue_completeness(self):
        """All catalogue entries have required fields."""
        required = {"endpoint", "method", "param", "type", "difficulty", "severity"}
        for vuln_id, vuln in VULNERABILITY_CATALOGUE.items():
            assert required <= set(vuln.keys()), f"Missing fields in {vuln_id}"


# ---------------------------------------------------------------------------
# Scenario tests
# ---------------------------------------------------------------------------

class TestScenarios:

    def test_all_scenarios_have_vulns_or_clean(self):
        for sc_id, scenario in SCENARIOS.items():
            assert hasattr(scenario, "active_vulns")
            assert hasattr(scenario, "safe_endpoints")
            assert scenario.step_budget > 0
            assert scenario.token_budget > 0

    def test_holdout_clean_has_zero_vulns(self):
        scenario = get_scenario("holdout_clean")
        assert len(scenario.active_vulns) == 0

    def test_train_full_has_all_vulns(self):
        scenario = get_scenario("train_full")
        assert len(scenario.active_vulns) == 5

    def test_scenario_split_labels(self):
        from src.environment.scenarios import get_train_scenarios, get_eval_scenarios, get_holdout_scenarios
        train = get_train_scenarios()
        eval_ = get_eval_scenarios()
        hold = get_holdout_scenarios()
        assert all(s.split == "train" for s in train)
        assert all(s.split == "eval" for s in eval_)
        assert all(s.split == "holdout" for s in hold)

    def test_get_unknown_scenario_raises(self):
        with pytest.raises(KeyError):
            get_scenario("does_not_exist")


# ---------------------------------------------------------------------------
# RL Environment tests (no live server needed — use mock)
# ---------------------------------------------------------------------------

class TestSQLiEnvironment:

    @pytest.fixture(scope="class")
    def env_and_scenario(self):
        env = SQLiEnvironment(port=5099)
        env.start_server()
        scenario = get_scenario("train_easy_single")
        return env, scenario

    def test_reset_returns_state(self, env_and_scenario):
        env, scenario = env_and_scenario
        state = env.reset(scenario)
        assert state is not None
        assert state.step == 0
        assert state.step_budget == scenario.step_budget
        assert len(state.findings) == 0

    def test_enumerate_endpoints_tool(self, env_and_scenario):
        env, scenario = env_and_scenario
        state = env.reset(scenario)
        action = {"tool": "enumerate_endpoints", "params": {}}
        next_state, reward, done, info = env.step(action, tokens_used=50)
        assert not done
        assert len(next_state.discovered_endpoints) > 1

    def test_inject_payload_tool(self, env_and_scenario):
        env, scenario = env_and_scenario
        state = env.reset(scenario)
        env.step({"tool": "enumerate_endpoints", "params": {}}, tokens_used=50)
        action = {
            "tool": "inject_payload",
            "params": {
                "endpoint": "/api/login",
                "method": "POST",
                "param": "username",
                "payload": "' OR '1'='1",
                "attack_type": "error_based",
            },
        }
        next_state, reward, done, info = env.step(action, tokens_used=100)
        result = info.get("result", {})
        assert "injection_signals" in result

    def test_report_finding_tool(self, env_and_scenario):
        env, scenario = env_and_scenario
        state = env.reset(scenario)
        action = {
            "tool": "report_finding",
            "params": {
                "endpoint": "/api/login",
                "param": "username",
                "payload": "' OR '1'='1",
                "vuln_type": "error_based",
                "confidence": 0.9,
                "evidence": "SQL error in response",
            },
        }
        next_state, reward, done, info = env.step(action, tokens_used=50)
        assert len(next_state.findings) == 1
        assert reward > 0

    def test_stop_tool_terminates_episode(self, env_and_scenario):
        env, scenario = env_and_scenario
        state = env.reset(scenario)
        action = {"tool": "stop", "params": {"reason": "testing_complete"}}
        next_state, reward, done, info = env.step(action, tokens_used=10)
        assert done
        assert next_state.termination_reason == "agent_stopped"

    def test_budget_exhaustion(self, env_and_scenario):
        env, scenario = env_and_scenario
        state = env.reset(scenario)
        for _ in range(scenario.step_budget):
            if state.episode_done:
                break
            action = {"tool": "http_request", "params": {"url": "/api/health", "method": "GET"}}
            state, _, done, _ = env.step(action, tokens_used=10)
        assert state.episode_done

    def test_episode_metrics_structure(self, env_and_scenario):
        env, scenario = env_and_scenario
        env.reset(scenario)
        env.step({"tool": "stop", "params": {}}, tokens_used=10)
        metrics = env.get_episode_metrics()
        required_keys = {"dcs", "stc_score", "ter", "avds", "hir", "derr", "tie"}
        assert required_keys <= set(metrics.keys())

    def test_false_positive_detection(self, env_and_scenario):
        env, scenario = env_and_scenario
        env.reset(scenario)
        action = {
            "tool": "report_finding",
            "params": {
                "endpoint": "/api/health",
                "param": "status",
                "payload": "' OR 1=1",
                "vuln_type": "error_based",
                "confidence": 0.9,
                "evidence": "Imagined error",
            },
        }
        _, reward, _, _ = env.step(action, tokens_used=10)
        assert reward < 0, "False positive must yield negative reward"

    def test_duplicate_report_penalized(self, env_and_scenario):
        env, scenario = env_and_scenario
        env.reset(scenario)
        finding_action = {
            "tool": "report_finding",
            "params": {
                "endpoint": "/api/login",
                "param": "username",
                "payload": "' OR '1'='1",
                "vuln_type": "error_based",
                "confidence": 0.9,
                "evidence": "error in body",
            },
        }
        _, reward1, _, _ = env.step(finding_action, tokens_used=10)
        _, reward2, _, _ = env.step(finding_action, tokens_used=10)
        assert reward2 < reward1, "Duplicate report must yield lower reward"
