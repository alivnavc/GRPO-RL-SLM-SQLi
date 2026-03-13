"""
GRPO training script for Qwen2.5-Coder-0.5B-Instruct on the SQLi pen test task.

Usage:
    python grpo_pipeline/scripts/train_grpo.py --epochs 10 --G 4
    python grpo_pipeline/scripts/train_grpo.py --epochs 5 --G 2 --port 5001

Phases:
  1. Start Flask vulnerable target (reuses existing src/environment/vulnerable_app.py)
  2. Load Qwen2.5-Coder-0.5B-Instruct with LoRA
  3. Pre-GRPO evaluation  → snapshot all KPIs
  4. GRPO training loop   → N epochs, G rollouts per scenario per epoch
  5. Post-GRPO evaluation → snapshot all KPIs
  6. Print full before/after comparison table
  7. Save snapshots to results/grpo/
"""

import argparse
import os
import sys
import time
import io

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.environment.sqli_env import SQLiEnvironment
from grpo_pipeline.agent.local_llm import LocalLLM
from grpo_pipeline.rl.grpo_trainer import GRPOConfig, GRPOTrainer
from grpo_pipeline.eval.evaluator import (
    GRPOEvaluator,
    print_comparison,
    save_snapshot,
)


def parse_args():
    p = argparse.ArgumentParser(description="GRPO training for SQLi agent")
    p.add_argument("--epochs", type=int, default=10, help="Number of GRPO epochs")
    p.add_argument("--G", type=int, default=4, help="Rollouts per scenario per epoch")
    p.add_argument("--lr", type=float, default=2e-5, help="AdamW learning rate")
    p.add_argument("--beta", type=float, default=0.01, help="KL penalty coefficient")
    p.add_argument("--port", type=int, default=5001, help="Flask target port")
    p.add_argument("--model", type=str, default="Qwen/Qwen2.5-Coder-0.5B-Instruct")
    p.add_argument("--eval-episodes", type=int, default=2, help="Episodes per scenario during eval")
    p.add_argument("--output-dir", type=str, default="results/grpo")
    p.add_argument("--checkpoint-dir", type=str, default="results/grpo_checkpoints")
    p.add_argument("--verbose", action="store_true", help="Print each action+payload during eval episodes")
    p.add_argument("--log-file", type=str, default="", help="Also write all output to this file (tee)")
    return p.parse_args()


def _print_epoch_row(epoch: int, result: dict) -> None:
    print(
        f"  Epoch {epoch:3d} | loss={result['mean_loss']:.4f} | "
        f"reward={result['mean_reward']:6.2f} "
        f"[{result['min_reward']:.1f} … {result['max_reward']:.1f}]"
    )


class _Tee:
    """Write to both stdout and a file simultaneously."""
    def __init__(self, path: str):
        self._file = open(path, "w", buffering=1, encoding="utf-8")
        self._stdout = sys.stdout

    def write(self, data):
        self._stdout.write(data)
        self._file.write(data)

    def flush(self):
        self._stdout.flush()
        self._file.flush()

    def close(self):
        self._file.close()
        sys.stdout = self._stdout


def main():
    args = parse_args()

    tee = None
    if args.log_file:
        tee = _Tee(args.log_file)
        sys.stdout = tee
        print(f"[Logger] Writing all output to {args.log_file}")

    try:
        from rich.console import Console
        from rich.panel import Panel
        Console().print(
            Panel(
                f"[bold cyan]GRPO Training — SQLi Agent[/bold cyan]\n"
                f"Model: {args.model}\n"
                f"Epochs: {args.epochs}  |  G={args.G}  |  lr={args.lr}  |  β={args.beta}",
                title="GRPO Pipeline"
            )
        )
    except ImportError:
        print(f"=== GRPO Training | model={args.model} epochs={args.epochs} G={args.G} ===")

    # ------------------------------------------------------------------
    # Phase 1: Start environment
    # ------------------------------------------------------------------
    print("\nPhase 1: Starting vulnerable target application...")
    env = SQLiEnvironment(port=args.port)
    env.start_server()
    time.sleep(1.5)

    # ------------------------------------------------------------------
    # Phase 2: Load model
    # ------------------------------------------------------------------
    print("\nPhase 2: Loading Qwen2.5-Coder-0.5B-Instruct with LoRA...")
    llm = LocalLLM(model_name=args.model)

    config = GRPOConfig(
        G=args.G,
        beta=args.beta,
        lr=args.lr,
        checkpoint_dir=args.checkpoint_dir,
    )
    trainer = GRPOTrainer(llm=llm, env=env, config=config)
    evaluator = GRPOEvaluator(llm=llm, env=env, episodes_per_scenario=args.eval_episodes)

    # ------------------------------------------------------------------
    # Phase 3: Pre-GRPO evaluation
    # ------------------------------------------------------------------
    print("\nPhase 3: Pre-GRPO evaluation (epoch 0 — untrained weights)...")
    pre_snap = evaluator.evaluate(label="pre_grpo_epoch_0", verbose=args.verbose)
    save_snapshot(pre_snap, os.path.join(args.output_dir, "pre_grpo.json"))

    # ------------------------------------------------------------------
    # Phase 4: GRPO training
    # ------------------------------------------------------------------
    print(f"\nPhase 4: GRPO Training ({args.epochs} epochs, G={args.G} rollouts/scenario)...")
    print(f"{'Epoch':>7}  {'Loss':>8}  {'MeanReward':>12}  {'Range':>15}")
    print("-" * 50)

    for epoch in range(1, args.epochs + 1):
        result = trainer.train_epoch(epoch)
        _print_epoch_row(epoch, result)

        # Mid-training snapshot every 5 epochs
        if epoch % 5 == 0:
            mid_snap = evaluator.evaluate(label=f"mid_grpo_epoch_{epoch}", verbose=args.verbose)
            save_snapshot(mid_snap, os.path.join(args.output_dir, f"mid_epoch_{epoch:03d}.json"))
            print(f"  → Mid-eval: DCS={mid_snap.dcs:.3f}  AFS={mid_snap.afs:.3f}")

    # ------------------------------------------------------------------
    # Phase 5: Post-GRPO evaluation
    # ------------------------------------------------------------------
    print(f"\nPhase 5: Post-GRPO evaluation (after epoch {args.epochs})...")
    post_snap = evaluator.evaluate(label=f"post_grpo_epoch_{args.epochs}", verbose=args.verbose)
    save_snapshot(post_snap, os.path.join(args.output_dir, "post_grpo.json"))

    # ------------------------------------------------------------------
    # Phase 6: Full before/after comparison
    # ------------------------------------------------------------------
    print("\nPhase 6: Full KPI comparison — Before vs After GRPO")
    print_comparison(pre_snap, post_snap)

    # ------------------------------------------------------------------
    # Phase 7: Regression check
    # ------------------------------------------------------------------
    regressions = []
    for attr, threshold in [("dcs", 0.05), ("afs", 0.05), ("stc_score", 0.05)]:
        delta = getattr(post_snap, attr) - getattr(pre_snap, attr)
        if delta < -threshold:
            regressions.append(f"{attr}: {delta:+.3f}")

    if regressions:
        print(f"\n⚠  Regressions detected: {', '.join(regressions)}")
    else:
        print(f"\n✅  No regressions detected. DCS delta: {post_snap.dcs - pre_snap.dcs:+.3f}")

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    env.stop_server()
    print(f"\nDone. Results saved to {args.output_dir}/")
    print(f"LoRA checkpoints in {args.checkpoint_dir}/")
    if args.log_file:
        print(f"Full log saved to {args.log_file}")
    if tee:
        tee.close()


if __name__ == "__main__":
    main()
