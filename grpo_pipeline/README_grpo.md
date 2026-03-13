# GRPO Pipeline — Weight-Level RL for SQLi Agent

This folder contains a self-contained GRPO (Group Relative Policy Optimization) training
pipeline that fine-tunes `Qwen2.5-Coder-0.5B-Instruct` on the SQL injection pen testing task.
It **reuses the existing Flask environment and reward function** from `src/` without modifying them.

---

## Quick Start

```bash
# Install GRPO-specific dependencies (in addition to main requirements.txt)
pip install -r grpo_pipeline/requirements_grpo.txt

# Run full pipeline: pre-eval → train 10 epochs → post-eval → comparison
python grpo_pipeline/scripts/train_grpo.py --epochs 10 --G 4

# Faster smoke test (2 epochs, G=2 rollouts)
python grpo_pipeline/scripts/train_grpo.py --epochs 2 --G 2
```

**System requirements:** 12+ GB RAM (30 GB recommended for G=4), 24-core CPU (~2 hrs for 10 epochs).

---

## Architecture

```
grpo_pipeline/
├── agent/
│   └── local_llm.py        # Qwen2.5-Coder-0.5B-Instruct + LoRA (PEFT)
│                            # generate() — inference, no grad
│                            # compute_log_probs() — teacher-forcing, keeps grad
├── rl/
│   └── grpo_trainer.py     # GRPO training loop
│                            # _run_episode()    — collect rollout
│                            # _compute_grpo_loss() — advantages + KL penalty
│                            # GRPOTrainer.train_epoch() — one full pass
├── eval/
│   └── evaluator.py        # ALL 12 KPIs from kpi_framework.md
│                            # GRPOEvaluator.evaluate() → KPISnapshot
│                            # print_comparison() → before/after table
└── scripts/
    └── train_grpo.py       # CLI entry point (6 phases)
```

Connects to existing codebase:
- `src/environment/sqli_env.py` — same RL environment (unchanged)
- `src/environment/vulnerable_app.py` — same Flask target (unchanged)
- `src/reward/reward_function.py` — same KPI→reward translation (unchanged)
- `src/agent/prompts.py` — same system prompts (unchanged)

---

## GRPO Algorithm

### Why GRPO over alternatives

| Algorithm | Why not used here |
|-----------|-------------------|
| PPO | Requires separate value/critic network — doubles memory. GRPO uses group statistics as baseline instead. |
| DPO | Offline — needs pre-collected preference pairs. GRPO is online, updating weights as the agent interacts with the environment. |
| RLHF | Requires human raters. Our reward is fully automated from the environment. |
| Bandit (main pipeline) | No weight updates — only selects between prompt variants. GRPO actually changes what the model knows. |

**Why GRPO fits pen testing specifically:**
DeepSeek-R1 used GRPO for math because the reward is *verifiable* — answers are right or wrong.
SQL injection detection is identically verifiable: either the endpoint returns a SQL error and gets
marked injectable, or it doesn't. No human judgement needed. This makes GRPO's automatic reward
signal directly applicable.

### Algorithm in one paragraph

For each training step, sample a scenario and run **G=4 complete episodes** with the current
policy. Score each episode's total reward using the existing KPI reward function. Compute
advantages by normalising rewards within the group: `advantage_i = (r_i - mean(r)) / (std(r) + ε)`.
Recompute log probabilities of all generated tokens via teacher-forcing (so gradients flow back
through the LoRA adapters). The GRPO loss is:

```
L = -mean_G( advantage_i × mean_tokens( log_p_θ(t) - β × (log_p_θ(t) - log_p_ref(t)) ) )
```

The `β × KL` term penalises deviation from the frozen reference model (base Qwen weights),
preventing the policy from drifting into degenerate outputs. `β=0.01` is a light regulariser —
enough to stabilise training without suppressing exploration.

---

## KPIs Tracked (Before and After)

All 12 metrics from `kpi_framework.md` are computed per evaluation run:

| KPI | Category | Formula | Target |
|-----|----------|---------|--------|
| DCS | Effectiveness | `TP/known - 2×FP/reports` | ≥ 0.75 |
| True Positives | Effectiveness | mean TPs per episode | max |
| False Positives | Effectiveness | mean FPs per episode | = 0 |
| STC Score | Efficiency | `1 - (steps-optimal)/budget` | ≥ 0.60 |
| TER | Efficiency | `DCS / (tokens/1K)` | ≥ 0.05 |
| TIE | Efficiency | `productive_calls/total_calls` | ≥ 0.60 |
| HIR | Autonomy | `interventions/decisions` (always 0) | ≤ 0.05 |
| DERR | Autonomy | `recoveries/dead_ends` | ≥ 0.70 |
| STA | Autonomy | `1 - (premature_stops+overruns)/eps` | ≥ 0.80 |
| AVDS | Diversity | `unique_attack_types/applicable` | ≥ 0.50 |
| AFS | Composite | `0.4×DCS + 0.25×STC + 0.2×(1-HIR) + 0.15×AVDS` | ≥ 0.60 |
| Mean Reward | RL Signal | sum of step + terminal rewards | max |

### Expected improvement trajectory

| Epoch | DCS | AFS | Description |
|-------|-----|-----|-------------|
| 0 (untrained) | ~0.20 | ~0.35 | Qwen flails — generates malformed JSON, wrong endpoints |
| 3 | ~0.45 | ~0.50 | Learns enumerate-first pattern, occasional correct inject |
| 7 | ~0.70 | ~0.62 | Consistent error-based detection, begins reporting correctly |
| 10 | ~0.85 | ~0.72 | Systematic: enumerate → inject → report, low FPs |

These are estimates. Actual values depend on random seed and model download version.

---

## Regression Detection

After training, the script automatically checks for regressions:
```
DCS delta < -0.05 → regression flagged
AFS delta < -0.05 → regression flagged
STC delta < -0.05 → regression flagged
```

Mid-training snapshots are saved every 5 epochs to `results/grpo/mid_epoch_NNN.json`
so you can plot the learning curve and detect if training diverged.

---

## Limitations

1. **CPU speed** — 10 epochs with G=4 takes ~2 hours on 24 cores. Reduce to `--G 2 --epochs 5`
   for a faster smoke test (~30 min).

2. **Single environment** — Training and eval both use the same Flask app. In production,
   rotate across DVWA / Juice Shop with EDR normalisation (kpi_framework.md §B.3).

3. **0.5B model capacity** — Qwen2.5-Coder-0.5B is sufficient to learn the ReAct format
   and basic SQLi patterns but will plateau before reaching human-expert DCS. Upgrade to
   1.5B for higher ceiling.

4. **LoRA rank=16** — Conservative. If training loss plateaus early, increase `r` to 32
   in `local_llm.py:LORA_CFG`.
