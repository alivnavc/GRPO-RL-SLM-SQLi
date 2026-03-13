"""
Tool definitions and parser for the ReAct agent.

Tools are the agent's action space. Each tool has:
  - a name (used in the LLM output)
  - a parameter schema (validated with Pydantic)
  - a description used in the system prompt
"""

import json
import re
from typing import Any, Dict, Optional, Tuple

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

class HttpRequestParams(BaseModel):
    url: str = Field(..., description="Full URL or path (e.g. /api/login)")
    method: str = Field(default="GET", description="HTTP method")
    params: Dict[str, str] = Field(default_factory=dict, description="Query parameters")
    data: Dict[str, Any] = Field(default_factory=dict, description="POST body as JSON")
    headers: Dict[str, str] = Field(default_factory=dict)

    @field_validator("method")
    @classmethod
    def uppercase_method(cls, v):
        return v.upper()


class InjectPayloadParams(BaseModel):
    endpoint: str = Field(..., description="Target endpoint path, e.g. /api/login")
    method: str = Field(default="GET")
    param: str = Field(..., description="Parameter name to inject into")
    payload: str = Field(..., description="The SQLi payload string")
    attack_type: str = Field(
        default="unknown",
        description="One of: error_based, union_based, blind_boolean, blind_time, stacked",
    )

    @field_validator("method")
    @classmethod
    def uppercase_method(cls, v):
        return v.upper()


class ReportFindingParams(BaseModel):
    endpoint: str = Field(..., description="Vulnerable endpoint path")
    param: str = Field(..., description="Vulnerable parameter name")
    payload: str = Field(..., description="Payload that confirmed the vulnerability")
    vuln_type: str = Field(..., description="Vulnerability type: error_based, union_based, etc.")
    confidence: float = Field(default=0.8, ge=0.0, le=1.0, description="Confidence score 0-1")
    evidence: str = Field(..., description="Snippet from response that confirms the vuln")


class AnalyzeResponseParams(BaseModel):
    body: str = Field(..., description="Response body to analyze")
    status_code: int = Field(default=200)
    response_time_ms: int = Field(default=0)


class EnumerateEndpointsParams(BaseModel):
    pass


class StopParams(BaseModel):
    reason: str = Field(default="testing_complete", description="Why the agent is stopping")


TOOL_SCHEMAS = {
    "http_request": HttpRequestParams,
    "inject_payload": InjectPayloadParams,
    "report_finding": ReportFindingParams,
    "analyze_response": AnalyzeResponseParams,
    "enumerate_endpoints": EnumerateEndpointsParams,
    "stop": StopParams,
}


# ---------------------------------------------------------------------------
# LLM output parser
# ---------------------------------------------------------------------------

class ParsedAction:
    """Result of parsing one ReAct step from LLM output."""

    def __init__(
        self,
        thought: str,
        tool: str,
        params: Dict[str, Any],
        raw: str,
        parse_error: Optional[str] = None,
    ):
        self.thought = thought
        self.tool = tool
        self.params = params
        self.raw = raw
        self.parse_error = parse_error

    @property
    def is_valid(self) -> bool:
        return self.parse_error is None

    def to_env_action(self) -> Dict[str, Any]:
        return {"tool": self.tool, "params": self.params}


def parse_llm_output(text: str) -> ParsedAction:
    """
    Parse LLM output in ReAct format:

        Thought: <reasoning>
        Action: <tool_name>
        Params: <json>

    Returns a ParsedAction with validation errors if parsing fails.
    """
    thought_match = re.search(r"Thought:\s*(.+?)(?=Action:|$)", text, re.DOTALL | re.IGNORECASE)
    action_match = re.search(r"Action:\s*(\w+)", text, re.IGNORECASE)
    params_match = re.search(r"Params:\s*(\{.*?\}|\[.*?\])", text, re.DOTALL | re.IGNORECASE)

    thought = thought_match.group(1).strip() if thought_match else ""
    tool = action_match.group(1).strip().lower() if action_match else ""
    params_raw = params_match.group(1).strip() if params_match else "{}"

    if not tool:
        return ParsedAction(
            thought=thought,
            tool="",
            params={},
            raw=text,
            parse_error="No Action found in LLM output",
        )

    try:
        params = json.loads(params_raw)
    except json.JSONDecodeError as exc:
        params = _fuzzy_parse_params(params_raw)
        if params is None:
            return ParsedAction(
                thought=thought,
                tool=tool,
                params={},
                raw=text,
                parse_error=f"Invalid JSON in Params: {exc}",
            )

    schema_cls = TOOL_SCHEMAS.get(tool)
    if schema_cls is None:
        return ParsedAction(
            thought=thought,
            tool=tool,
            params=params,
            raw=text,
            parse_error=f"Unknown tool: {tool!r}",
        )

    try:
        validated = schema_cls(**params)
        params = validated.model_dump()
    except Exception as exc:
        return ParsedAction(
            thought=thought,
            tool=tool,
            params=params,
            raw=text,
            parse_error=f"Param validation error: {exc}",
        )

    return ParsedAction(thought=thought, tool=tool, params=params, raw=text)


def _fuzzy_parse_params(raw: str) -> Optional[Dict]:
    """
    Attempt to recover malformed JSON by fixing common LLM formatting mistakes.
    Returns None if irrecoverable.
    """
    raw = raw.strip()
    raw = re.sub(r",\s*}", "}", raw)
    raw = re.sub(r",\s*]", "]", raw)
    raw = re.sub(r"'", '"', raw)
    raw = re.sub(r"(\w+):", r'"\1":', raw)
    try:
        return json.loads(raw)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Token counting (approximate, without tiktoken dependency for offline use)
# ---------------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token (GPT-3.5 average)."""
    return max(1, len(text) // 4)
