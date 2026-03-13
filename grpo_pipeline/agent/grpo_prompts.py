"""
Few-shot-enhanced system prompt for Qwen2.5-Coder-0.5B-Instruct.

Small models (< 1B params) need explicit in-context examples to follow
a structured output format reliably. Without these, the model generates
natural prose that fails the Thought/Action/Params parser, which falls back
to stop, ending the episode in 1 step with reward = step_cost, std=0,
GRPO loss=0, and no gradient update.

These examples show the exact format expected by parse_llm_output() in
src/agent/tools.py.
"""

FEW_SHOT_EXAMPLES = (
    "\n---\n"
    "IMPORTANT RULES:\n"
    "- /api/info only returns a list of endpoints. It has NO database interaction. NEVER inject into /api/info.\n"
    "- Injectable endpoints are those that accept user input: /api/login, /api/search, /api/products/<id>, /api/orders.\n"
    "- Use a single quote (') as payload to trigger SQL syntax errors (status 500).\n\n"
    "EXAMPLE INTERACTION (follow this format exactly for every response):\n\n"
    "Step 1: Discover endpoints\n"
    "Thought: I need to discover all available endpoints before testing anything.\n"
    "Action: enumerate_endpoints\n"
    'Params: {}\n\n'
    "Step 2: Inject into /api/login (NOT /api/info — that has no DB)\n"
    "Thought: /api/info is discovery-only. I will test /api/login username with a single quote to trigger a SQL error.\n"
    "Action: inject_payload\n"
    'Params: {"endpoint": "/api/login", "method": "POST", "param": "username", '
    '"payload": "\'", "attack_type": "error_based"}\n\n'
    "Step 3: Report after seeing sql_error in last result\n"
    "Thought: The response contained a SQL error confirming injection in /api/login username.\n"
    "Action: report_finding\n"
    'Params: {"endpoint": "/api/login", "param": "username", "payload": "\'", '
    '"vuln_type": "error_based", "confidence": 0.9, "evidence": "SQL error in response"}\n\n'
    "Step 4: Test another endpoint\n"
    "Thought: I will also test /api/search q parameter with a single quote.\n"
    "Action: inject_payload\n"
    'Params: {"endpoint": "/api/search", "method": "GET", "param": "q", '
    '"payload": "\'", "attack_type": "error_based"}\n\n'
    "Step 5: Stop when done\n"
    "Thought: All injection points have been tested. Assessment complete.\n"
    "Action: stop\n"
    'Params: {"reason": "testing_complete"}\n'
    "---\n"
    "Now follow the EXACT same Thought/Action/Params format for every response. "
    "Never inject into /api/info. Never deviate from this format.\n"
)


def build_grpo_system_prompt(base_system_prompt: str) -> str:
    """
    Append few-shot examples to the base system prompt.
    Called by LocalLLM.generate() to ensure Qwen follows the ReAct format.
    """
    return base_system_prompt + FEW_SHOT_EXAMPLES

