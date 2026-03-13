"""
Tests for the ReAct agent, tool parser, and bandit policy.
"""

import json
import pytest

from src.agent.tools import parse_llm_output, estimate_tokens, TOOL_SCHEMAS
from src.agent.prompts import build_system_prompt, build_task_prompt, get_all_strategy_ids
from src.agent.react_agent import LLMBackend, ReActAgent
from src.rl.policy import UCBBandit
from src.rl.trajectory_buffer import TrajectoryBuffer, Episode
from src.environment.scenarios import get_scenario


# ---------------------------------------------------------------------------
# Tool parser tests
# ---------------------------------------------------------------------------

class TestToolParser:

    def test_parse_valid_stop(self):
        text = 'Thought: Done testing.\nAction: stop\nParams: {"reason": "complete"}'
        parsed = parse_llm_output(text)
        assert parsed.is_valid
        assert parsed.tool == "stop"
        assert parsed.params["reason"] == "complete"
        assert "Done testing" in parsed.thought

    def test_parse_inject_payload(self):
        params = {
            "endpoint": "/api/login",
            "method": "POST",
            "param": "username",
            "payload": "' OR '1'='1",
            "attack_type": "error_based",
        }
        text = f"Thought: Testing login.\nAction: inject_payload\nParams: {json.dumps(params)}"
        parsed = parse_llm_output(text)
        assert parsed.is_valid
        assert parsed.tool == "inject_payload"
        assert parsed.params["param"] == "username"
        assert parsed.params["attack_type"] == "error_based"

    def test_parse_report_finding(self):
        params = {
            "endpoint": "/api/login",
            "param": "username",
            "payload": "' OR '1'='1",
            "vuln_type": "error_based",
            "confidence": 0.9,
            "evidence": "SQL error: no such column",
        }
        text = f"Thought: Confirmed.\nAction: report_finding\nParams: {json.dumps(params)}"
        parsed = parse_llm_output(text)
        assert parsed.is_valid
        assert parsed.params["confidence"] == 0.9

    def test_parse_missing_action(self):
        text = "Thought: I should test something."
        parsed = parse_llm_output(text)
        assert not parsed.is_valid
        assert "No Action" in parsed.parse_error

    def test_parse_unknown_tool(self):
        text = "Thought: Test.\nAction: unknown_tool\nParams: {}"
        parsed = parse_llm_output(text)
        assert not parsed.is_valid
        assert "Unknown tool" in parsed.parse_error

    def test_parse_invalid_json_fuzzy_recovery(self):
        text = "Thought: Testing.\nAction: stop\nParams: {reason: 'done'}"
        parsed = parse_llm_output(text)

    def test_parse_enumerate_endpoints_no_params(self):
        text = "Thought: Discover endpoints.\nAction: enumerate_endpoints\nParams: {}"
        parsed = parse_llm_output(text)
        assert parsed.is_valid
        assert parsed.tool == "enumerate_endpoints"

    def test_to_env_action(self):
        text = "Thought: Stop.\nAction: stop\nParams: {}"
        parsed = parse_llm_output(text)
        action = parsed.to_env_action()
        assert action["tool"] == "stop"
        assert isinstance(action["params"], dict)

    def test_method_uppercased(self):
        params = {
            "endpoint": "/api/login",
            "method": "post",
            "param": "username",
            "payload": "test",
        }
        text = f"Thought: x\nAction: inject_payload\nParams: {json.dumps(params)}"
        parsed = parse_llm_output(text)
        assert parsed.is_valid
        assert parsed.params["method"] == "POST"

    def test_estimate_tokens(self):
        assert estimate_tokens("hello world") > 0
        assert estimate_tokens("a" * 400) == pytest.approx(100, abs=10)


# ---------------------------------------------------------------------------
# Prompt tests
# ---------------------------------------------------------------------------

class TestPrompts:

    def test_all_strategies_buildable(self):
        for strat_id in get_all_strategy_ids():
            prompt = build_system_prompt(strat_id)
            assert len(prompt) > 100
            assert "AVAILABLE TOOLS" in prompt

    def test_default_strategy_fallback(self):
        prompt = build_system_prompt("nonexistent_strategy")
        assert len(prompt) > 100

    def test_task_prompt_includes_state(self):
        from src.environment.sqli_env import AgentState, EndpointInfo
        state = AgentState(
            step=3,
            step_budget=20,
            token_budget=8000,
            tokens_used=500,
            discovered_endpoints=[EndpointInfo(url="/api/login", method="POST", params=["username"])],
            findings=[],
            last_tool_call="inject_payload",
            last_tool_result={"status_code": 500},
            dead_end_streak=0,
        )
        prompt = build_task_prompt(state.to_prompt_context(), "Test scenario")
        assert "Step 3/20" in prompt
        assert "/api/login" in prompt


# ---------------------------------------------------------------------------
# Mock LLM backend tests
# ---------------------------------------------------------------------------

class TestMockLLMBackend:

    def test_mock_backend_returns_string(self):
        llm = LLMBackend(provider="mock")
        text, tokens = llm.complete("system", "user message about /api/login")
        assert isinstance(text, str)
        assert tokens > 0

    def test_mock_backend_enumerate_first(self):
        llm = LLMBackend(provider="mock")
        text, _ = llm.complete("system", "Step 0/20")
        assert "enumerate_endpoints" in text.lower() or "action" in text.lower()

    def test_mock_backend_stop_fallback(self):
        llm = LLMBackend(provider="mock")
        text, _ = llm.complete("system", "I have tested everything and nothing is left")
        assert isinstance(text, str)


# ---------------------------------------------------------------------------
# ReAct agent tests
# ---------------------------------------------------------------------------

class TestReActAgent:

    @pytest.fixture(scope="class")
    def agent(self):
        llm = LLMBackend(provider="mock")
        return ReActAgent(llm=llm)

    def test_agent_reset_clears_state(self, agent):
        agent._history = [{"role": "assistant", "content": "x"}]
        agent._total_tokens = 999
        agent.reset()
        assert agent._history == []
        assert agent._total_tokens == 0

    def test_agent_reset_changes_strategy(self, agent):
        agent.reset(strategy_id="aggressive_fuzzer")
        assert agent.strategy_id == "aggressive_fuzzer"

    def test_agent_full_episode_with_mock(self):
        from src.environment.sqli_env import SQLiEnvironment
        env = SQLiEnvironment(port=5098)
        env.start_server()
        scenario = get_scenario("train_easy_single")
        llm = LLMBackend(provider="mock")
        agent = ReActAgent(llm=llm)
        metrics, trajectory = agent.run_episode(env, scenario, verbose=False)

        assert "dcs" in metrics
        assert "total_reward" in metrics
        assert isinstance(trajectory, list)
        assert len(trajectory) > 0

    def test_agent_tokens_tracked(self):
        from src.environment.sqli_env import SQLiEnvironment
        env = SQLiEnvironment(port=5097)
        env.start_server()
        scenario = get_scenario("train_easy_single")
        llm = LLMBackend(provider="mock")
        agent = ReActAgent(llm=llm)
        agent.run_episode(env, scenario)
        assert agent.total_tokens > 0


# ---------------------------------------------------------------------------
# UCB Bandit tests
# ---------------------------------------------------------------------------

class TestUCBBandit:

    def test_select_untried_arms_first(self):
        bandit = UCBBandit()
        seen = set()
        for _ in range(len(bandit.arms)):
            arm = bandit.select("test_scenario")
            bandit.update("test_scenario", arm, 0.5)
            seen.add(arm)
        assert seen == set(bandit.arms)

    def test_best_arm_tracking(self):
        bandit = UCBBandit()
        for arm in bandit.arms:
            bandit.update("s1", arm, 0.3)
        bandit.update("s1", "evidence_first", 0.9)
        best, score = bandit.best_arm("s1")
        assert best == "evidence_first"

    def test_convergence_after_many_pulls(self):
        bandit = UCBBandit(exploration_constant=0.5)
        for _ in range(30):
            arm = bandit.select("s2")
            reward = 0.9 if arm == "systematic_methodical" else 0.1
            bandit.update("s2", arm, reward)
        conv = bandit.convergence_summary("s2")
        assert conv["best_arm"] == "systematic_methodical"

    def test_save_and_load(self, tmp_path):
        bandit = UCBBandit()
        bandit.update("s1", "systematic_methodical", 0.8)
        bandit.update("s1", "aggressive_fuzzer", 0.4)
        save_path = str(tmp_path / "bandit.json")
        bandit.save(save_path)
        loaded = UCBBandit.load(save_path)
        assert loaded._counts["s1"]["systematic_methodical"] == 1
        assert loaded._rewards["s1"]["aggressive_fuzzer"] == pytest.approx(0.4)

    def test_arm_stats_structure(self):
        bandit = UCBBandit()
        bandit.update("s1", "systematic_methodical", 0.7)
        stats = bandit.get_arm_stats("s1")
        assert "systematic_methodical" in stats
        assert "pulls" in stats["systematic_methodical"]
        assert "mean_reward" in stats["systematic_methodical"]


# ---------------------------------------------------------------------------
# Trajectory buffer tests
# ---------------------------------------------------------------------------

class TestTrajectoryBuffer:

    @pytest.fixture
    def scenario(self):
        return get_scenario("train_easy_single")

    def test_add_episode(self, scenario):
        buf = TrajectoryBuffer()
        metrics = {"dcs": 0.8, "stc_score": 0.7, "avds": 0.5, "derr": 1.0, "tie": 0.6,
                   "false_positives": 0, "findings_count": 1}
        ep = buf.add("train_easy_single", "systematic_methodical", metrics, [], scenario)
        assert ep.episode_id is not None
        assert ep.score > 0

    def test_strategy_stats_accuracy(self, scenario):
        buf = TrajectoryBuffer()
        metrics_good = {"dcs": 0.9, "stc_score": 0.8, "avds": 0.75, "derr": 1.0, "tie": 0.8,
                        "false_positives": 0, "findings_count": 1}
        metrics_bad = {"dcs": 0.2, "stc_score": 0.1, "avds": 0.0, "derr": 0.0, "tie": 0.2,
                       "false_positives": 2, "findings_count": 2}
        buf.add("train_easy_single", "systematic_methodical", metrics_good, [], scenario)
        buf.add("train_easy_single", "aggressive_fuzzer", metrics_bad, [], scenario)
        good_stats = buf.get_strategy_stats("train_easy_single", "systematic_methodical")
        bad_stats = buf.get_strategy_stats("train_easy_single", "aggressive_fuzzer")
        assert good_stats["mean"] > bad_stats["mean"]

    def test_preference_pairs_generated(self, scenario):
        buf = TrajectoryBuffer()
        metrics_a = {"dcs": 0.9, "stc_score": 0.8, "avds": 0.75, "derr": 1.0, "tie": 0.8,
                     "false_positives": 0, "findings_count": 1}
        metrics_b = {"dcs": 0.3, "stc_score": 0.4, "avds": 0.25, "derr": 0.5, "tie": 0.4,
                     "false_positives": 1, "findings_count": 2}
        buf.add("train_easy_single", "systematic_methodical", metrics_a, [], scenario)
        buf.add("train_easy_single", "aggressive_fuzzer", metrics_b, [], scenario)
        pairs = buf.build_preference_pairs("train_easy_single", min_gap=0.05)
        assert len(pairs) >= 1
        assert pairs[0].chosen.strategy_id == "systematic_methodical"

    def test_total_episodes_count(self, scenario):
        buf = TrajectoryBuffer()
        metrics = {"dcs": 0.5, "stc_score": 0.5, "avds": 0.5, "derr": 1.0, "tie": 0.5,
                   "false_positives": 0, "findings_count": 1}
        for _ in range(5):
            buf.add("train_easy_single", "systematic_methodical", metrics, [], scenario)
        assert buf.total_episodes() == 5

    def test_persist_and_reload(self, tmp_path, scenario):
        save_dir = str(tmp_path)
        buf = TrajectoryBuffer(save_dir=save_dir)
        metrics = {"dcs": 0.7, "stc_score": 0.6, "avds": 0.5, "derr": 1.0, "tie": 0.7,
                   "false_positives": 0, "findings_count": 1}
        buf.add("train_easy_single", "systematic_methodical", metrics, [], scenario)
        buf2 = TrajectoryBuffer()
        buf2.load_from_dir(save_dir)
        assert buf2.total_episodes() == 1
