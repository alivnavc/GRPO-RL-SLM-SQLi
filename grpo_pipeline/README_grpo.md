# GRPO Pipeline — Weight-Level RL for SQLi Agent

Fine-tunes `Qwen2.5-Coder-0.5B-Instruct` via LoRA on the SQL injection pen testing task using GRPO. Reuses the Flask environment and reward function from `src/`.

---

## Quick Start

```bash
pip install -r grpo_pipeline/requirements_grpo.txt
python grpo_pipeline/scripts/train_grpo.py --epochs 10 --G 4 --verbose --log-file grpo_training.log

# Smoke test
python grpo_pipeline/scripts/train_grpo.py --epochs 2 --G 2
```

**GPU recommended** (8+ GB VRAM). CPU fallback works but is slow. Model loads in `bfloat16` automatically.

---

## Structure

```
grpo_pipeline/
├── agent/
│   ├── local_llm.py        # Qwen2.5-Coder-0.5B-Instruct + LoRA
│   │                        # generate() — inference only
│   │                        # compute_log_probs() — teacher-forcing for GRPO loss
│   └── grpo_prompts.py     # Few-shot examples (enumerate → inject → report)
│                            # Injects into every system prompt for format compliance
├── rl/
│   └── grpo_trainer.py     # _run_episode() — collect rollout
│                            # _compute_grpo_loss() — group advantages + KL penalty
│                            # GRPOTrainer.train_epoch() — AdamW LoRA update
├── eval/
│   └── evaluator.py        # 12 KPIs, before/after snapshot, comparison table
└── scripts/
    └── train_grpo.py       # 7-phase CLI (--epochs, --G, --lr, --beta, --verbose, --log-file)
```

Connects to `src/`:
- `sqli_env.py` — shared RL environment (`enumerate_endpoints` reward fixed to count only newly discovered endpoints)
- `vulnerable_app.py` — shared Flask target (request/response logging hooks added)
- `reward_function.py` — shared KPI → reward translation (unchanged)

---

## How It Works

1. **Start** Flask target on port 5001
2. **Load** Qwen with LoRA in bfloat16 + frozen reference copy
3. **Pre-eval** — snapshot all 12 KPIs before training
4. **Train** — each epoch: run G=4 episodes per scenario, compute group advantages, backprop through LoRA
5. **Mid-eval** — snapshot every 5 epochs → `results/grpo/mid_epoch_NNN.json`
6. **Post-eval** — snapshot after final epoch
7. **Compare** — print before/after table, flag regressions (DCS/AFS/STC delta < -0.05)

**Agent flow per step:** model generates `Thought → Action → Params` → parsed by `src/agent/tools.py` → executed against Flask → reward computed → StepRecord stored (only if action was not overridden by guard).

**Memory optimisations:** `bfloat16`, gradient checkpointing, `PYTORCH_ALLOC_CONF=expandable_segments:True`, `MAX_SEQ_LEN=768` truncation, `del logits` after use, `gc.collect()` + `empty_cache()` after each optimizer step.
