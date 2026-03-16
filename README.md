# Agent Performance Engineering for Web App Pen Testing
## SQL Injection — RL POC

A proof-of-concept reinforcement learning pipeline that measures and systematically improves an AI agent's performance on SQL injection penetration testing tasks.

---

## Quick Start

### Bandit pipeline (no GPU needed)

```bash
cp .env.example .env
pip install -r requirements.txt
python scripts/demo.py
```

### GRPO pipeline (Qwen local model, no API key needed)

```bash
pip install -r requirements.txt
pip install -r grpo_pipeline/requirements_grpo.txt
python grpo_pipeline/scripts/train_grpo.py --epochs 10 --G 4 --verbose --log-file grpo_training.log
```

See [`grpo_pipeline/README_grpo.md`](grpo_pipeline/README_grpo.md) for GRPO-specific docs.

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
├── grpo_pipeline/                # Weight-level RL (GRPO fine-tuning)
│   ├── agent/
│   │   ├── local_llm.py          # Qwen2.5-Coder-0.5B-Instruct + LoRA (PEFT)
│   │   └── grpo_prompts.py       # Few-shot examples for ReAct format compliance
│   ├── rl/
│   │   └── grpo_trainer.py       # GRPO loop: rollout → advantage → loss → backprop
│   ├── eval/
│   │   └── evaluator.py          # All 12 KPIs, before/after snapshot + comparison
│   ├── scripts/
│   │   └── train_grpo.py         # CLI: 7-phase pipeline with --verbose/--log-file
│   ├── requirements_grpo.txt
│   └── README_grpo.md
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

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Flask Target App                         │
│   /api/login  /api/search  /api/products/<id>  /api/orders      │
│   (5 SQLi vulnerabilities across difficulty tiers 1–4)          │
└──────────────────────────┬──────────────────────────────────────┘
                           │ HTTP
          ┌────────────────┴────────────────┐
          │                                 │
┌─────────▼──────────┐           ┌──────────▼──────────┐
│   Bandit Pipeline  │           │   GRPO Pipeline     │
│   (src/)           │           │   (grpo_pipeline/)  │
│                    │           │                     │
│  ReAct Agent       │           │  Qwen2.5-Coder-0.5B │
│  ↓                 │           │  + LoRA adapters    │
│  UCB1 Bandit       │           │  ↓                  │
│  (5 prompt arms)   │           │  G=4 rollouts/epoch │
│  ↓                 │           │  ↓                  │
│  KPI scoring       │           │  GRPO loss          │
│  ↓                 │           │  ↓                  │
│  Bandit update     │           │  AdamW LoRA update  │
└─────────┬──────────┘           └──────────┬──────────┘
          │                                 │
          └────────────────┬────────────────┘
                           │
              ┌────────────▼────────────┐
              │     Shared components   │
              │  sqli_env.py  (RL env)  │
              │  reward_function.py     │
              │  scenarios.py           │
              └─────────────────────────┘
```

**Key design choices:**
- Both pipelines share the same Flask target, RL environment, and reward function — only the learning mechanism differs.
- Bandit: no weight updates, selects best prompt strategy at inference time.
- GRPO: actual gradient updates to LoRA weights via teacher-forcing log prob recomputation.
- Agent format: `Thought → Action → Params` (ReAct), parsed by `src/agent/tools.py`.

---

## How It Works

### Target environment

A mock Flask app (`src/environment/vulnerable_app.py`) with 5 intentional SQL injection points across different endpoints and difficulty tiers. The agent interacts with it over HTTP using 6 tools: `enumerate_endpoints`, `http_request`, `inject_payload`, `analyze_response`, `report_finding`, `stop`.

### Bandit pipeline (`src/`)

The agent runs ReAct episodes (Thought → Action → Observation) against the Flask target. A UCB1 bandit selects which of 5 prompt strategies to use each episode, then updates based on the episode's reward score. No model weights are changed — only the prompt strategy is selected at inference time.

```bash
python scripts/train.py --episodes 50
python scripts/evaluate.py run --split eval
python scripts/evaluate.py compare results/eval/before.json results/eval/after.json
python scripts/evaluate.py full-pipeline --provider mock
```

### GRPO pipeline (`grpo_pipeline/`)

Fine-tunes `Qwen2.5-Coder-0.5B-Instruct` via LoRA using GRPO. Runs G=4 full episodes per scenario per epoch, scores each by total reward, computes advantages within the group, and backpropagates through the LoRA adapters. Saves a before/after KPI snapshot for comparison.

```bash
python grpo_pipeline/scripts/train_grpo.py --epochs 10 --G 4 --verbose --log-file grpo_training.log
```

### Tests

```bash
pytest tests/ -v
```

### Docker

```bash
docker compose run demo
docker compose run train
docker compose run test
```
