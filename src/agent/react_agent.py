"""
ReAct Agent for SQL Injection penetration testing.

Implements the Thought → Action → Observation loop using an LLM backend.
Supports OpenAI and Anthropic. Falls back to a deterministic rule-based
agent for testing without an API key.
"""

import os
import json
import time
import logging
from typing import Any, Dict, Generator, List, Optional, Tuple

from src.agent.prompts import build_system_prompt, build_task_prompt, DEFAULT_STRATEGY
from src.agent.tools import parse_llm_output, estimate_tokens, ParsedAction
from src.environment.sqli_env import SQLiEnvironment, AgentState
from src.environment.scenarios import Scenario

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM backend abstraction
# ---------------------------------------------------------------------------

class LLMBackend:
    """Thin wrapper around LLM API calls."""

    def __init__(self, provider: str = "openai", model: Optional[str] = None):
        self.provider = provider.lower()
        self.model = model or self._default_model()
        self._client = None

    def _default_model(self) -> str:
        defaults = {
            "openai": "gpt-4o-mini",
            "anthropic": "claude-3-haiku-20240307",
            "mock": "mock-deterministic",
        }
        return defaults.get(self.provider, "gpt-4o-mini")

    def _get_client(self):
        if self._client:
            return self._client
        if self.provider == "openai":
            from openai import OpenAI
            self._client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        elif self.provider == "anthropic":
            import anthropic
            self._client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        return self._client

    def complete(self, system_prompt: str, user_message: str, max_tokens: int = 512) -> Tuple[str, int]:
        """
        Returns (response_text, tokens_used).
        """
        if self.provider == "mock":
            return self._mock_complete(user_message)

        try:
            if self.provider == "openai":
                return self._openai_complete(system_prompt, user_message, max_tokens)
            elif self.provider == "anthropic":
                return self._anthropic_complete(system_prompt, user_message, max_tokens)
        except Exception as exc:
            logger.warning(f"LLM call failed ({exc}), falling back to mock")
            return self._mock_complete(user_message)

        return self._mock_complete(user_message)

    def _openai_complete(self, system: str, user: str, max_tokens: int) -> Tuple[str, int]:
        client = self._get_client()
        resp = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
            temperature=0.2,
        )
        text = resp.choices[0].message.content or ""
        tokens = resp.usage.total_tokens if resp.usage else estimate_tokens(system + user + text)
        return text, tokens

    def _anthropic_complete(self, system: str, user: str, max_tokens: int) -> Tuple[str, int]:
        client = self._get_client()
        resp = client.messages.create(
            model=self.model,
            system=system,
            messages=[{"role": "user", "content": user}],
            max_tokens=max_tokens,
        )
        text = resp.content[0].text if resp.content else ""
        tokens = (resp.usage.input_tokens + resp.usage.output_tokens) if resp.usage else estimate_tokens(system + user + text)
        return text, tokens

    def reset_mock(self):
        """Reset mock internal state for a new episode."""
        self._mock_plan_idx = 0
        self._mock_pending_report: Optional[Tuple[str, str]] = None

    def _mock_complete(self, user_message: str) -> Tuple[str, int]:
        """
        Stateful step-driven mock that follows an explicit ordered plan.
        Tracks its own position so it never re-tests an endpoint.
        """
        import re

        # Initialise on first call
        if not hasattr(self, "_mock_plan_idx"):
            self.reset_mock()

        # Check if the LAST RESULT (current state block) has injection signals
        last_result_block = ""
        if "last result:" in user_message.lower():
            last_result_block = user_message.lower().split("last result:")[-1].split("\n")[0]
        has_signals = "sql_error" in last_result_block or (
            "injection_signals" in last_result_block and "500" in last_result_block
        )

        # If there are pending signals from the last injection → report finding
        if has_signals and self._mock_pending_report:
            endpoint, param = self._mock_pending_report
            self._mock_pending_report = None
            params_r = json.dumps({
                "endpoint": endpoint, "param": param, "payload": "'",
                "vuln_type": "error_based", "confidence": 0.9,
                "evidence": "SQL error confirmed in response",
            })
            response = (
                f"Thought: SQL error confirms injection at {endpoint} param={param}.\n"
                f"Action: report_finding\nParams: {params_r}"
            )
            return response, estimate_tokens(response)

        # Advance through plan
        plan = self._build_mock_plan(user_message)
        idx = self._mock_plan_idx

        if idx >= len(plan):
            response = (
                "Thought: Assessment complete. All planned injection tests done.\n"
                "Action: stop\n"
                'Params: {"reason": "testing_complete"}'
            )
            return response, estimate_tokens(response)

        action_fn = plan[idx]
        self._mock_plan_idx += 1
        return action_fn()

    def _build_mock_plan(self, user_message: str):
        """Build ordered list of callables representing the mock's test plan."""
        plan = []

        # Step 1: enumerate
        def do_enumerate():
            r = ("Thought: Enumerate endpoints first.\n"
                 "Action: enumerate_endpoints\nParams: {}")
            return r, estimate_tokens(r)

        plan.append(do_enumerate)

        # Only add injection steps if those endpoints appear in the message
        if "/api/login" in user_message:
            def do_inject_login():
                self._mock_pending_report = ("/api/login", "username")
                p = json.dumps({"endpoint": "/api/login", "method": "POST",
                                "param": "username", "payload": "'",
                                "attack_type": "error_based"})
                r = f"Thought: Probe /api/login for error-based SQLi.\nAction: inject_payload\nParams: {p}"
                return r, estimate_tokens(r)
            plan.append(do_inject_login)

        if "/api/products/search" in user_message:
            def do_inject_search():
                self._mock_pending_report = ("/api/products/search", "q")
                p = json.dumps({"endpoint": "/api/products/search", "method": "GET",
                                "param": "q", "payload": "'",
                                "attack_type": "error_based"})
                r = f"Thought: Probe /api/products/search for SQLi.\nAction: inject_payload\nParams: {p}"
                return r, estimate_tokens(r)
            plan.append(do_inject_search)

        if "/api/orders" in user_message:
            def do_inject_orders():
                self._mock_pending_report = ("/api/orders", "status")
                p = json.dumps({"endpoint": "/api/orders", "method": "GET",
                                "param": "status", "payload": "pending' AND '1'='1",
                                "attack_type": "blind_boolean"})
                r = f"Thought: Probe /api/orders for blind boolean SQLi.\nAction: inject_payload\nParams: {p}"
                return r, estimate_tokens(r)
            plan.append(do_inject_orders)

        def do_stop():
            r = ("Thought: All targets tested. Stopping.\n"
                 "Action: stop\nParams: {\"reason\": \"testing_complete\"}")
            return r, estimate_tokens(r)

        plan.append(do_stop)
        return plan

    def _extract_last_inject_target(self, text: str) -> tuple:
        """Parse the last inject_payload Params block to get endpoint+param."""
        import re, json as _json
        blocks = re.findall(
            r'Action:\s*inject_payload\s*\nParams:\s*(\{[^\n]+\})', text
        )
        for block in reversed(blocks):
            try:
                p = _json.loads(block)
                return p.get("endpoint", "/api/unknown"), p.get("param", "unknown")
            except Exception:
                continue
        matches = re.findall(r'/api/[^\s"\'>\]]+', text)
        return (matches[-1] if matches else "/api/unknown"), "unknown"

    def _extract_last_endpoint(self, text: str) -> str:
        return self._extract_last_inject_target(text)[0]

    def _extract_last_param(self, text: str) -> str:
        return self._extract_last_inject_target(text)[1]


# ---------------------------------------------------------------------------
# ReAct Agent
# ---------------------------------------------------------------------------

class ReActAgent:
    """
    SQLi penetration testing agent using the ReAct (Reason + Act) paradigm.

    The agent maintains its own conversation history and calls the LLM
    to produce one Thought+Action per step. The environment executes the
    action and returns an observation, which is appended to the history.
    """

    def __init__(
        self,
        llm: Optional[LLMBackend] = None,
        strategy_id: str = DEFAULT_STRATEGY,
        max_retries: int = 2,
    ):
        self.llm = llm or LLMBackend(provider="mock")
        self.strategy_id = strategy_id
        self.max_retries = max_retries
        self._history: List[Dict[str, str]] = []
        self._total_tokens = 0

    def reset(self, strategy_id: Optional[str] = None):
        """Reset agent state for a new episode."""
        if strategy_id:
            self.strategy_id = strategy_id
        self._history = []
        self._total_tokens = 0

    @property
    def total_tokens(self) -> int:
        return self._total_tokens

    def act(self, state: AgentState, scenario: Scenario) -> Tuple[ParsedAction, int]:
        """
        Given the current environment state, produce the next action.

        Returns:
            (ParsedAction, tokens_used_this_step)
        """
        system_prompt = build_system_prompt(self.strategy_id)
        task_prompt = build_task_prompt(
            state_context=state.to_prompt_context(),
            scenario_description=scenario.description,
        )

        if self._history:
            context = "\n\n".join(
                f"[Step {i+1}]\n{msg['content']}"
                for i, msg in enumerate(self._history[-6:])
            )
            user_message = context + "\n\n" + task_prompt
        else:
            user_message = task_prompt

        for attempt in range(self.max_retries + 1):
            raw_output, tokens = self.llm.complete(
                system_prompt=system_prompt,
                user_message=user_message,
                max_tokens=512,
            )
            self._total_tokens += tokens
            parsed = parse_llm_output(raw_output)

            if parsed.is_valid:
                self._history.append(
                    {"role": "assistant", "content": raw_output}
                )
                return parsed, tokens

            if attempt < self.max_retries:
                retry_msg = (
                    f"Your last output had a parsing error: {parsed.parse_error}\n"
                    "Please retry with the correct format:\n"
                    "Thought: <reasoning>\nAction: <tool_name>\nParams: <json>"
                )
                user_message = user_message + "\n\n" + retry_msg
                logger.debug(f"Parse error on attempt {attempt+1}: {parsed.parse_error}")

        logger.warning("All parse attempts failed; defaulting to stop action")
        fallback = ParsedAction(
            thought="Parsing failed repeatedly; stopping.",
            tool="stop",
            params={"reason": "parse_failure"},
            raw="",
        )
        return fallback, tokens

    def run_episode(
        self, env: SQLiEnvironment, scenario: Scenario, verbose: bool = False
    ) -> Tuple[Dict[str, Any], List[Dict]]:
        """
        Run a complete episode: reset → loop until done → return metrics + trajectory.

        Returns:
            (episode_metrics, trajectory)
            trajectory = list of step dicts with state, action, reward, info
        """
        self.reset()
        state = env.reset(scenario)
        trajectory = []
        total_reward = 0.0

        while not state.episode_done:
            parsed_action, tokens = self.act(state, scenario)
            env_action = parsed_action.to_env_action()
            next_state, reward, done, info = env.step(env_action, tokens_used=tokens)

            step_record = {
                "step": state.step,
                "thought": parsed_action.thought,
                "tool": parsed_action.tool,
                "params": parsed_action.params,
                "reward": reward,
                "injection_signals": info.get("result", {}).get("injection_signals", []),
                "productive": info.get("productive", False),
            }
            trajectory.append(step_record)
            total_reward += reward

            if verbose:
                logger.info(
                    f"Step {state.step}: {parsed_action.tool} | reward={reward:.3f} | "
                    f"findings={len(next_state.findings)}"
                )

            state = next_state

        metrics = env.get_episode_metrics()
        metrics["total_reward"] = round(total_reward, 4)
        metrics["strategy_id"] = self.strategy_id

        return metrics, trajectory
