# Agent Performance Engineering for Web App Pen Testing
## SQL Injection — RL POC

A proof-of-concept reinforcement learning pipeline that measures and systematically improves an AI agent's performance on SQL injection penetration testing tasks.

---

## Quick Start

```bash
# No API key needed — runs with mock LLM
cp .env.example .env
pip install -r requirements.txt

# Run the full demo (pre-eval → train → post-eval → comparison)
python scripts/demo.py

# Or with Docker
docker compose run demo
```

---

## Project Structure

```
browserAgentFox/
├── kpi_framework.md              # Part 1: Full KPI design document
├── src/
│   ├── environment/
│   │   ├── vulnerable_app.py     # Mock Flask app with 5 SQLi injection points
│   │   ├── sqli_env.py           # RL environment (state, action space, step)
│   │   └── scenarios.py          # Train/eval/holdout scenario definitions
│   ├── agent/
│   │   ├── react_agent.py        # ReAct agent (Thought → Action → Observation)
│   │   ├── tools.py              # Tool schemas + LLM output parser
│   │   └── prompts.py            # 5 strategy variants (bandit arms)
│   ├── reward/
│   │   └── reward_function.py    # KPI → reward signal translation
│   ├── rl/
│   │   ├── policy.py             # UCB1 Multi-Armed Bandit
│   │   ├── trajectory_buffer.py  # Episode storage + preference pairs
│   │   └── trainer.py            # Training loop
│   └── eval/
│       ├── harness.py            # Evaluation runner + before/after comparison
│       └── metrics.py            # KPI aggregation + regression detection
├── scripts/
│   ├── demo.py                   # Full pipeline demo (Rich UI)
│   ├── train.py                  # Training CLI
│   └── evaluate.py               # Evaluation CLI (run / compare / full-pipeline)
├── tests/
│   ├── test_environment.py
│   ├── test_reward.py
│   └── test_agent.py
├── Dockerfile
└── docker-compose.yml
```

---

## Part 1: KPI Framework

See [`kpi_framework.md`](kpi_framework.md) for the full design document.

**5 pen test tasks defined:**
1. SQL Injection Point Identification & Exploitation ← *RL POC task*
2. Authentication Bypass (credential stuffing + logic flaws)
3. IDOR (Insecure Direct Object Reference)
4. API Endpoint Enumeration from Swagger/OpenAPI
5. Session Management & JWT Security Analysis

**Key KPIs for Task 1 (SQLi):**

| KPI | Formula | Target |
|-----|---------|--------|
| DCS | `TP/known - λ*FP/reports` (λ=2.0) | ≥ 0.75 |
| STC_score | `1 - (steps - optimal) / budget` | ≥ 0.60 |
| TER | `DCS / (tokens / 1K)` | ≥ 0.05 |
| AVDS | `unique_attack_types / applicable_types` | ≥ 0.50 |
| HIR | `interventions / decisions` | ≤ 0.05 |
| DERR | `recoveries / dead_ends` | ≥ 0.70 |
| AFS | Weighted mean of task composites | ≥ 0.60 |

---

## Part 2: RL Implementation

### Environment

**Target:** Mock Flask application (`src/environment/vulnerable_app.py`) with **5 intentional SQLi vulnerabilities** at difficulty tiers 1–4:

| Vuln ID | Endpoint | Param | Type | Difficulty |
|---------|----------|-------|------|------------|
| login_username | `POST /api/login` | username | error_based | 1 |
| search_query | `GET /api/products/search` | q | union_based | 2 |
| product_id | `GET /api/products/<id>` | id | error_based | 1 |
| order_filter | `GET /api/orders` | status | blind_boolean | 3 |
| user_profile | `GET /api/users/<id>/profile` | fields | blind_time | 4 |

**State representation:** `AgentState` — discovered endpoints, findings, injection signals, dead-end streak, budget remaining.

**Action space:** 6 tools — `enumerate_endpoints`, `http_request`, `inject_payload`, `analyze_response`, `report_finding`, `stop`.

### Reward Function

The reward is designed to be **sparse + shaped** to avoid credit assignment problems without enabling reward hacking:

```
Per-step:
  - Step cost:              -0.10  (forces efficiency)
  - SQL error signal:       +1.50  (productive exploration signal)
  - New endpoint found:     +0.50
  - True positive report:   +7.0 × confidence_scale
  - False positive report:  -3.0 × confidence_scale  (2× the TP reward)
  - Dead-end penalty:       -0.20  (beyond step cost)

Terminal (episode-end):
  - DCS bonus:              +10.0 × DCS
  - STC bonus:              +3.0  × STC_score
  - Attack diversity bonus: +1.0   (if AVDS ≥ 0.5)
  - Perfect episode:        +5.0   (DCS≥0.9, FP=0)
```

**Why false positive penalty > true positive reward:** Reward hacking prevention. An agent that reports everything as vulnerable would accumulate large TP rewards — the FP penalty (2× TP) makes this strategy strictly worse than targeted testing.

**Why step cost:** Without it, agents enumerate endlessly. With it, the agent is incentivized to reach conclusions efficiently.

### RL Approach: Multi-Armed Bandit (UCB1)

**Choice:** UCB1 Bandit over 5 prompt strategy variants.

**Alternatives considered:**
- **PPO over token logits:** Requires differentiable policy + model weights. Infeasible for black-box LLM APIs (GPT-4o-mini, Claude).
- **RLHF:** Requires human raters. We automate this with KPI-scored trajectories (auto-RLHF).
- **DPO on ranked trajectories:** Valid complement. The `TrajectoryBuffer` builds preference pairs for offline DPO fine-tuning. Not the primary training signal here because it requires model weight access.
- **Bandit over strategies:** Directly optimizable, interpretable, production-ready (strategy = prompt config at inference time).

**The 5 bandit arms (strategy variants):**

| Strategy | Description |
|----------|-------------|
| `systematic_methodical` | Enumerate first, then test easy→hard |
| `aggressive_fuzzer` | Spray many payloads rapidly, breadth-first |
| `evidence_first` | Triple-check before reporting (low FP) |
| `blind_specialist` | Expert at boolean/time-based blind injection |
| `chain_exploiter` | Exploit fully before moving to next target |

**UCB1 selection:**
```
score(arm) = mean_reward(arm) + C × √(ln(N) / n(arm))
C = 1.41 (√2)
```

Untried arms are always selected first (infinite UCB score). After all arms tried, UCB1 balances exploration/exploitation. Over 30+ episodes per scenario, the bandit concentrates on the strategy with highest DCS × efficiency.

### Training Loop

```
for episode in range(num_episodes):
    scenario = sample_train_scenario()
    strategy = bandit.select(scenario.id)        # UCB1 selection
    metrics, traj = agent.run_episode(strategy)   # ReAct loop
    score = score_trajectory(metrics, traj)       # KPI → scalar
    bandit.update(scenario.id, strategy, score)   # UCB update
    buffer.add(episode)                           # Store for DPO pairs
```

### Evaluation & Regression Detection

**Eval harness** runs N episodes per scenario across eval/holdout splits and computes all Part-1 KPIs.

**Holdout scenarios include:**
- `holdout_clean` — zero vulnerabilities. Any finding = reward hacking / false positive.
- `holdout_partial` — novel param combination not seen in training.

**Regression detection** flags a regression when any critical metric drops >5pp from baseline:
```
python scripts/evaluate.py compare results/eval/before.json results/eval/after.json
```

---

## Usage

### Training

```bash
# Mock LLM (no key, deterministic)
python scripts/train.py --episodes 50

# OpenAI
python scripts/train.py --provider openai --episodes 100

# Specific scenarios only
python scripts/train.py --scenarios train_easy_single train_medium_two --episodes 30
```

### Evaluation

```bash
# Eval split with default strategy (pre-training baseline)
python scripts/evaluate.py run --split eval

# Eval with trained bandit
python scripts/evaluate.py run --split eval --bandit-path results/training/bandit_final.json

# Holdout evaluation (regression check)
python scripts/evaluate.py run --split holdout --bandit-path results/training/bandit_final.json

# Compare two eval reports
python scripts/evaluate.py compare results/eval/eval_before.json results/eval/eval_after.json

# Full pipeline in one command
python scripts/evaluate.py full-pipeline --provider mock
```

### Tests

```bash
pytest tests/ -v
pytest tests/test_environment.py -v    # Environment + Flask app
pytest tests/test_reward.py -v         # Reward function
pytest tests/test_agent.py -v          # Agent + bandit + buffer
```

### Docker

```bash
docker compose run demo       # Full demo
docker compose run train      # Training only
docker compose run evaluate   # Evaluation only
docker compose run test       # Test suite
```

---

## LLM Provider Notes

| Provider | Model Used | Why |
|----------|-----------|-----|
| `mock` | Deterministic rule-based | No API key; for CI/testing/demo |
| `openai` | `gpt-4o-mini` | Best cost/performance for tool-calling |
| `anthropic` | `claude-3-haiku-20240307` | Strong instruction following, lower cost |

**Model selection rationale:** GPT-4o-mini was chosen as the default real-LLM option because it has the best balance of instruction-following quality, JSON output reliability (critical for tool call parsing), and cost efficiency at ~$0.15/1M input tokens. Claude Haiku is the Anthropic alternative with similar tradeoffs.

The mock backend is deterministic and sufficient to validate the RL infrastructure without any API cost.

---

## What Worked / What Didn't

### What Worked

1. **Reward shaping via injection signals** — providing intermediate rewards for SQL error detection (not just final findings) dramatically reduced the sparse reward problem. The agent gets signal within 2–3 steps of testing an injectable parameter.

2. **False positive penalty > true positive reward** — cleanly prevents the most obvious reward hack (report everything). No further anti-gaming needed for the basic case.

3. **UCB1 over strategies** — clean, measurable convergence. After 30 episodes, the bandit concentrates on the best strategy and the before/after improvement is clearly attributable to strategy selection, not noise.

4. **Holdout clean scenario** — critical for catching reward hackers. An agent optimized only on training scenarios without this check would miss false-positive accumulation.

### What Didn't / Limitations

1. **Mock LLM doesn't explore much** — the deterministic mock backend follows a fixed script, so the bandit's arm diversity is limited. With a real LLM (GPT-4o-mini), the same-strategy variance is higher and UCB exploration is more meaningful.

2. **Blind injection reward is thin** — the blind_boolean and blind_time scenarios are harder to reward-shape because the signals (response count difference, timing) require multiple steps to confirm. The current reward function detects time delay but doesn't do differential boolean comparison automatically. A future improvement is an `observe_diff` tool that computes response deltas.

3. **No true fine-tuning** — the bandit selects strategy prompts but doesn't update model weights. For a production system, the `PreferencePair` objects in `TrajectoryBuffer` would feed a DPO fine-tuning pipeline on a locally hosted model (e.g., Llama-3).

4. **Single-target environment** — training on one Flask app risks overfitting to its specific schema. Production would require rotating target environments (DVWA, Juice Shop, HackTheBox machines) with EDR normalization as described in `kpi_framework.md`.

---

## What I'd Do With Another Week

1. **GRPO weight-level training** — Replace the UCB bandit with a full GRPO loop over `Qwen2.5-Coder-0.5B-Instruct` (LoRA, CPU-feasible on 30 GB RAM). The `TrajectoryBuffer` already builds preference pairs; GRPO would turn those into real gradient updates. Expected: DCS improves from ~0.3 → ~0.85 over 50 episodes as the model learns to generate targeted payloads instead of random ones.

2. **Differential boolean reward for blind SQLi** — The current reward detects time-delay signals but doesn't do automated response-diff comparison for blind boolean injection. Add an `observe_diff` tool that runs true/false condition pairs and computes a `response_delta` score. This unlocks the `blind_boolean` and `blind_time` vulnerability tiers in the Flask app that currently go undetected.

3. **Rotate training environments** — The `holdout_clean` scenario catches reward hacking but training on one Flask app risks overfitting to its specific schema. Integrate DVWA or OWASP Juice Shop as secondary training targets with EDR normalization (as defined in `kpi_framework.md` §B.3) so the agent learns general SQLi principles, not app-specific patterns.

4. **HCS (Hypothesis Consistency Score)** — The KPI framework defines this metric (CoT quality scored by an LLM judge) but it's not yet computed in the eval harness. Add a lightweight judge prompt that scores each Thought step 0–3 against the evidence received, normalized to [0,1]. This catches agents that produce correct answers via incorrect reasoning — the hardest form of reward hacking to detect.

5. **Extend to 3 more task types** — The KPI framework defines 5 tasks; only SQLi is implemented. Auth bypass (`/api/login` bypass via logic flaws) and IDOR (`/api/orders` parameter tampering) are already in the Flask app. Adding scenarios for these + shared reward function would demonstrate cross-task portfolio KPIs (AFS, skill gap matrix) with real data.

---

## Reference Codebase Awareness

This implementation is informed by the architecture patterns in:
- **CAI** — ReAct loop structure, human-in-the-loop hooks (HIR metric), tool-based action space
- **Strix** — Graph-based orchestration (our scenario DAG), sandboxed execution (Flask in-process)
- **Cyber-AutoAgent** — Memory (trajectory buffer), meta-agent reasoning (strategy selection), measurable improvement loops

Key differences: We prioritize **measurable improvement over task completion**. The reward function is directly derived from the KPI framework (not post-hoc), and the training loop is designed so every component is independently testable.
