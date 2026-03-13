"""
RL Environment wrapper for SQLi penetration testing.

Implements a Gymnasium-compatible interface around the mock Flask app.
Manages episode lifecycle, state representation, and step transitions.
"""

import json
import time
import threading
import requests
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

from src.environment.scenarios import Scenario, VulnPoint
from src.environment.vulnerable_app import create_app, VULNERABILITY_CATALOGUE


# ---------------------------------------------------------------------------
# State representation
# ---------------------------------------------------------------------------

@dataclass
class EndpointInfo:
    """Discovered endpoint with observed properties."""
    url: str
    method: str
    params: List[str]
    injectable_params: List[str] = field(default_factory=list)
    tested: bool = False


@dataclass
class Finding:
    """A confirmed or suspected injection point."""
    endpoint: str
    param: str
    payload: str
    vuln_type: str          # confirmed type or "suspected"
    confidence: float       # 0.0 - 1.0
    evidence: str           # snippet from response that supports the finding
    step_found: int = 0


@dataclass
class AgentState:
    """
    Full observable state passed to the agent at each step.

    Design rationale: We give the agent the minimum information needed
    to make a decision — discovered endpoints, what has been tested,
    current findings, and budget remaining. Crucially, we do NOT give
    it the ground truth vulnerability list (that would be cheating).
    """
    step: int
    step_budget: int
    token_budget: int
    tokens_used: int
    discovered_endpoints: List[EndpointInfo]
    findings: List[Finding]
    last_tool_call: Optional[str]
    last_tool_result: Optional[Dict[str, Any]]
    dead_end_streak: int       # consecutive steps with no new info
    episode_done: bool = False
    termination_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_prompt_context(self) -> str:
        """Serialize state to a compact string for the agent's context window."""
        endpoints_summary = []
        for ep in self.discovered_endpoints:
            tested_mark = "✓" if ep.tested else "○"
            inj = f" [INJECTABLE: {', '.join(ep.injectable_params)}]" if ep.injectable_params else ""
            endpoints_summary.append(f"  {tested_mark} {ep.method} {ep.url} params={ep.params}{inj}")

        findings_summary = []
        for f in self.findings:
            findings_summary.append(
                f"  [{f.confidence:.0%} confidence] {f.endpoint} param={f.param} "
                f"type={f.vuln_type} | evidence: {f.evidence[:80]}"
            )

        last_result_str = ""
        if self.last_tool_result:
            signals = self.last_tool_result.get("injection_signals", [])
            status = self.last_tool_result.get("status_code", "")
            err = self.last_tool_result.get("error", "")
            if signals:
                last_result_str = f"LAST RESULT: status={status} injection_signals={signals}\n"
            elif err:
                last_result_str = f"LAST RESULT: error={err}\n"
            else:
                last_result_str = f"LAST RESULT: status={status}\n"

        return (
            f"=== CURRENT STATE (Step {self.step}/{self.step_budget}) ===\n"
            f"Tokens used: {self.tokens_used}/{self.token_budget}\n"
            f"Dead-end streak: {self.dead_end_streak}\n"
            + last_result_str + "\n"
            + f"DISCOVERED ENDPOINTS:\n" + "\n".join(endpoints_summary or ["  None yet"]) + "\n\n"
            + f"FINDINGS ({len(self.findings)}):\n" + "\n".join(findings_summary or ["  None yet"]) + "\n"
        )


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class SQLiEnvironment:
    """
    Manages the mock app server, episode lifecycle, and step execution.

    episode flow:
        env.reset(scenario) → state
        while not done:
            action = agent.act(state)
            state, reward, done, info = env.step(action)
        metrics = env.get_episode_metrics()
    """

    def __init__(self, base_url: Optional[str] = None, port: int = 5001):
        self._port = port
        self._base_url = base_url or f"http://127.0.0.1:{port}"
        self._server_thread: Optional[threading.Thread] = None
        self._flask_app = None

        self._scenario: Optional[Scenario] = None
        self._state: Optional[AgentState] = None

        self._start_time: float = 0.0
        self._human_interventions: int = 0
        self._dead_ends_encountered: int = 0
        self._dead_end_recoveries: int = 0
        self._unique_attack_types_used: set = set()
        self._tool_calls_total: int = 0
        self._tool_calls_productive: int = 0
        self._previously_seen_endpoints: set = set()
        self._previously_seen_params: set = set()

    # -----------------------------------------------------------------------
    # Server lifecycle
    # -----------------------------------------------------------------------

    def start_server(self):
        """Start the Flask dev server in a daemon thread."""
        if self._server_thread and self._server_thread.is_alive():
            return
        self._flask_app = create_app()
        self._server_thread = threading.Thread(
            target=lambda: self._flask_app.run(
                host="127.0.0.1", port=self._port, debug=False, use_reloader=False
            ),
            daemon=True,
        )
        self._server_thread.start()
        self._wait_for_server()

    def _wait_for_server(self, timeout: float = 10.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                requests.get(f"{self._base_url}/api/health", timeout=1)
                return
            except requests.RequestException:
                time.sleep(0.2)
        raise RuntimeError(f"Flask server did not start within {timeout}s")

    def stop_server(self):
        """Daemon thread stops automatically; no explicit stop needed."""
        pass

    # -----------------------------------------------------------------------
    # Episode management
    # -----------------------------------------------------------------------

    def reset(self, scenario: Scenario) -> AgentState:
        """Start a new episode with the given scenario."""
        self._scenario = scenario
        self._start_time = time.time()
        self._human_interventions = 0
        self._dead_ends_encountered = 0
        self._dead_end_recoveries = 0
        self._unique_attack_types_used = set()
        self._tool_calls_total = 0
        self._tool_calls_productive = 0
        self._previously_seen_endpoints = set()
        self._previously_seen_params = set()

        initial_endpoints = [
            EndpointInfo(
                url=f"{self._base_url}/api/info",
                method="GET",
                params=[],
            )
        ]

        self._state = AgentState(
            step=0,
            step_budget=scenario.step_budget,
            token_budget=scenario.token_budget,
            tokens_used=0,
            discovered_endpoints=initial_endpoints,
            findings=[],
            last_tool_call=None,
            last_tool_result=None,
            dead_end_streak=0,
        )
        return self._state

    def step(self, action: Dict[str, Any], tokens_used: int = 0) -> Tuple[AgentState, float, bool, Dict]:
        """
        Execute one agent action and return (new_state, reward, done, info).

        action format:
            {
                "tool": "http_request" | "inject_payload" | "report_finding"
                       | "analyze_response" | "stop",
                "params": { ... tool-specific ... }
            }
        """
        if self._state is None:
            raise RuntimeError("Call reset() before step()")

        self._state.step += 1
        self._state.tokens_used += tokens_used
        self._tool_calls_total += 1

        tool = action.get("tool", "")
        params = action.get("params", {})

        prev_findings_count = len(self._state.findings)
        prev_endpoints_count = len(self._state.discovered_endpoints)

        result = self._dispatch_tool(tool, params)
        self._state.last_tool_call = tool
        self._state.last_tool_result = result

        productive = self._is_productive(prev_findings_count, prev_endpoints_count)
        if productive:
            self._tool_calls_productive += 1
            if self._state.dead_end_streak >= 3:
                self._dead_end_recoveries += 1
            self._state.dead_end_streak = 0
        else:
            self._state.dead_end_streak += 1
            if self._state.dead_end_streak == 3:
                self._dead_ends_encountered += 1

        done, reason = self._check_termination(action)
        self._state.episode_done = done
        self._state.termination_reason = reason

        from src.reward.reward_function import compute_step_reward
        reward = compute_step_reward(
            state=self._state,
            action=action,
            result=result,
            scenario=self._scenario,
            productive=productive,
        )

        info = {
            "tool": tool,
            "productive": productive,
            "wall_time": time.time() - self._start_time,
            "result": result,
        }

        return self._state, reward, done, info

    # -----------------------------------------------------------------------
    # Tool dispatcher
    # -----------------------------------------------------------------------

    def _dispatch_tool(self, tool: str, params: Dict) -> Dict:
        dispatch = {
            "http_request": self._tool_http_request,
            "inject_payload": self._tool_inject_payload,
            "report_finding": self._tool_report_finding,
            "analyze_response": self._tool_analyze_response,
            "enumerate_endpoints": self._tool_enumerate_endpoints,
            "stop": self._tool_stop,
        }
        handler = dispatch.get(tool)
        if handler is None:
            return {"error": f"Unknown tool: {tool!r}", "valid_tools": list(dispatch.keys())}
        try:
            return handler(params)
        except Exception as exc:
            return {"error": str(exc)}

    def _tool_http_request(self, params: Dict) -> Dict:
        """Make an HTTP request to the target app."""
        url = params.get("url", "")
        method = params.get("method", "GET").upper()
        data = params.get("data", {})
        query_params = params.get("params", {})
        headers = params.get("headers", {})

        if not url.startswith(self._base_url):
            url = self._base_url + url if url.startswith("/") else url

        try:
            resp = requests.request(
                method, url,
                json=data if method in ("POST", "PUT", "PATCH") else None,
                params=query_params if method == "GET" else None,
                headers=headers,
                timeout=5,
            )
            response_body = resp.text[:2000]
            result = {
                "status_code": resp.status_code,
                "body": response_body,
                "headers": dict(resp.headers),
                "response_time_ms": int(resp.elapsed.total_seconds() * 1000),
            }
            self._update_discovered_endpoints(url, method, query_params, data)
            return result
        except requests.RequestException as exc:
            return {"error": str(exc)}

    def _tool_inject_payload(self, params: Dict) -> Dict:
        """
        Inject a SQLi payload into a specific parameter and return
        structured result with injection-specific metadata.
        """
        endpoint = params.get("endpoint", "")
        method = params.get("method", "GET").upper()
        param = params.get("param", "")
        payload = params.get("payload", "")
        attack_type = params.get("attack_type", "unknown")

        self._unique_attack_types_used.add(attack_type)

        url = self._base_url + endpoint if endpoint.startswith("/") else endpoint

        try:
            if method == "GET":
                resp = requests.get(url, params={param: payload}, timeout=5)
            else:
                resp = requests.post(url, json={param: payload}, timeout=5)

            body = resp.text[:2000]
            response_time_ms = int(resp.elapsed.total_seconds() * 1000)

            injection_signals = self._detect_injection_signals(body, resp.status_code, response_time_ms)

            self._update_param_tested(endpoint, param)

            return {
                "status_code": resp.status_code,
                "body": body,
                "response_time_ms": response_time_ms,
                "injection_signals": injection_signals,
                "payload_tested": payload,
                "param": param,
                "endpoint": endpoint,
            }
        except requests.RequestException as exc:
            return {"error": str(exc)}

    def _tool_report_finding(self, params: Dict) -> Dict:
        """Agent reports a confirmed or suspected vulnerability."""
        endpoint = params.get("endpoint", "")
        param = params.get("param", "")
        payload = params.get("payload", "")
        vuln_type = params.get("vuln_type", "unknown")
        confidence = float(params.get("confidence", 0.5))
        evidence = params.get("evidence", "")
        step_found = self._state.step

        existing_ids = {(f.endpoint, f.param) for f in self._state.findings}
        if (endpoint, param) not in existing_ids:
            finding = Finding(
                endpoint=endpoint,
                param=param,
                payload=payload,
                vuln_type=vuln_type,
                confidence=confidence,
                evidence=evidence,
                step_found=step_found,
            )
            self._state.findings.append(finding)
            return {"recorded": True, "finding_count": len(self._state.findings)}

        return {"recorded": False, "reason": "duplicate", "finding_count": len(self._state.findings)}

    def _tool_analyze_response(self, params: Dict) -> Dict:
        """
        Agent analyzes a previous response for injection signals.
        This is a 'thinking' step — no HTTP request made.
        """
        body = params.get("body", "")
        status_code = params.get("status_code", 200)
        response_time_ms = params.get("response_time_ms", 0)
        signals = self._detect_injection_signals(body, status_code, response_time_ms)
        return {"signals": signals, "analysis": "complete"}

    def _tool_enumerate_endpoints(self, params: Dict) -> Dict:
        """Fetch /api/info to discover endpoints."""
        try:
            resp = requests.get(f"{self._base_url}/api/info", timeout=5)
            data = resp.json()
            raw_endpoints = data.get("endpoints", [])

            prev_count = len(self._state.discovered_endpoints)

            for ep_path in raw_endpoints:
                method = "POST" if "login" in ep_path else "GET"
                ep_params = self._infer_params(ep_path)
                self._update_discovered_endpoints(
                    self._base_url + ep_path, method, {p: "" for p in ep_params}, {}
                )

            newly_added = [
                ep.url.replace(self._base_url, "")
                for ep in self._state.discovered_endpoints[prev_count:]
            ]

            return {
                "discovered": newly_added,
                "all_known": raw_endpoints,
                "total_known": len(self._state.discovered_endpoints),
            }
        except Exception as exc:
            return {"error": str(exc)}

    def _tool_stop(self, params: Dict) -> Dict:
        """Agent explicitly terminates the episode."""
        reason = params.get("reason", "agent_decision")
        return {"stopped": True, "reason": reason}

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _detect_injection_signals(self, body: str, status_code: int, response_time_ms: int) -> List[str]:
        signals = []
        body_lower = body.lower()

        sql_error_patterns = [
            "sqlite", "syntax error", "operational error", "sql",
            "near", "unrecognized token", "no such column", "no such table",
        ]
        for pattern in sql_error_patterns:
            if pattern in body_lower:
                signals.append(f"sql_error:{pattern}")

        if response_time_ms > 2000:
            signals.append("time_delay:>2s")
        if response_time_ms > 5000:
            signals.append("time_delay:>5s")

        if status_code == 500:
            signals.append("server_error:500")

        if "sqlite_master" in body_lower or "create table" in body_lower:
            signals.append("schema_exposure")

        if any(kw in body_lower for kw in ["admin", "sup3r", "alice123", "b0bpass"]):
            signals.append("credential_exposure")

        return signals

    def _update_discovered_endpoints(self, url: str, method: str, params: Dict, data: Dict):
        url_key = url.split("?")[0]
        if url_key in self._previously_seen_endpoints:
            return
        self._previously_seen_endpoints.add(url_key)

        all_params = list(params.keys()) + list(data.keys())
        ep = EndpointInfo(url=url_key, method=method, params=all_params)
        self._state.discovered_endpoints.append(ep)

    def _update_param_tested(self, endpoint: str, param: str):
        key = f"{endpoint}:{param}"
        if key not in self._previously_seen_params:
            self._previously_seen_params.add(key)
        for ep in self._state.discovered_endpoints:
            if endpoint in ep.url or ep.url in endpoint:
                ep.tested = True
                if param not in ep.injectable_params:
                    signals_in_last = self._state.last_tool_result or {}
                    if signals_in_last.get("injection_signals"):
                        ep.injectable_params.append(param)

    def _infer_params(self, path: str) -> List[str]:
        params = []
        if "<" in path:
            params.append("id")
        if "search" in path:
            params.append("q")
        if "orders" in path:
            params.append("status")
        if "profile" in path:
            params.append("fields")
        return params

    def _is_productive(self, prev_findings: int, prev_endpoints: int) -> bool:
        new_findings = len(self._state.findings) > prev_findings
        new_endpoints = len(self._state.discovered_endpoints) > prev_endpoints
        new_signals = bool(
            self._state.last_tool_result
            and self._state.last_tool_result.get("injection_signals")
        )
        return new_findings or new_endpoints or new_signals

    def _check_termination(self, action: Dict) -> Tuple[bool, str]:
        if action.get("tool") == "stop":
            return True, "agent_stopped"
        if self._state.step >= self._state.step_budget:
            return True, "budget_exhausted"
        if self._state.tokens_used >= self._state.token_budget:
            return True, "token_budget_exhausted"
        return False, ""

    # -----------------------------------------------------------------------
    # Post-episode metrics
    # -----------------------------------------------------------------------

    def get_episode_metrics(self) -> Dict[str, Any]:
        """
        Compute all Part-1 KPIs for the completed episode.
        Called by the reward function and eval harness.
        """
        if self._state is None or self._scenario is None:
            return {}

        known_vuln_ids = {v.vuln_id for v in self._scenario.active_vulns}
        known_count = len(known_vuln_ids)

        tp, fp = 0, 0
        for finding in self._state.findings:
            endpoint = finding.endpoint.replace(self._base_url, "")
            matched = any(
                v.endpoint.replace("{id}", "").rstrip("/") in endpoint
                and v.param == finding.param
                for v in self._scenario.active_vulns
            )
            if matched:
                tp += 1
            else:
                fp += 1

        lambda_fp = 2.0
        dcs = (tp / known_count - lambda_fp * (fp / max(len(self._state.findings), 1))
               if known_count > 0 else (0.0 if fp > 0 else 1.0))
        dcs = max(0.0, min(1.0, dcs))

        steps_optimal = max(known_count * 4, 8)
        stc_score = max(0.0, 1.0 - (self._state.step - steps_optimal) / self._state.step_budget)

        tokens_1k = max(self._state.tokens_used / 1000, 0.001)
        ter = dcs / tokens_1k

        applicable_attack_types = {"error_based", "union_based", "blind_boolean", "blind_time"}
        avds = len(self._unique_attack_types_used) / len(applicable_attack_types)

        hir = self._human_interventions / max(self._state.step, 1)
        derr = (self._dead_end_recoveries / self._dead_ends_encountered
                if self._dead_ends_encountered > 0 else 1.0)

        tie = self._tool_calls_productive / max(self._tool_calls_total, 1)

        wall_time = time.time() - self._start_time

        return {
            "dcs": round(dcs, 4),
            "true_positives": tp,
            "false_positives": fp,
            "known_vulns": known_count,
            "stc_score": round(stc_score, 4),
            "ter": round(ter, 4),
            "avds": round(avds, 4),
            "hir": round(hir, 4),
            "derr": round(derr, 4),
            "tie": round(tie, 4),
            "steps_taken": self._state.step,
            "tokens_used": self._state.tokens_used,
            "wall_time_s": round(wall_time, 2),
            "findings_count": len(self._state.findings),
            "dead_ends": self._dead_ends_encountered,
            "termination_reason": self._state.termination_reason,
        }
