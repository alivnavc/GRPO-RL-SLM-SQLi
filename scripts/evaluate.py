"""
Evaluation script: run the agent against eval/holdout scenarios and report KPIs.

Usage:
    # Evaluate using default strategy (no training)
    python scripts/evaluate.py

    # Evaluate using a trained bandit
    python scripts/evaluate.py --bandit-path results/training/bandit_final.json

    # Compare before/after training
    python scripts/evaluate.py --compare before.json after.json

    # Holdout evaluation (never-seen scenarios)
    python scripts/evaluate.py --split holdout --bandit-path results/training/bandit_final.json
"""

import logging
import os
import sys

import click

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.eval.harness import EvalConfig, EvalHarness, print_comparison

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("evaluate")


@click.group()
def cli():
    """Evaluation tools for the SQLi RL agent."""
    pass


@cli.command()
@click.option("--split", default="eval",
              type=click.Choice(["eval", "holdout", "train", "all"]),
              show_default=True, help="Which scenario split to evaluate on")
@click.option("--runs", default=3, show_default=True, help="Runs per scenario")
@click.option("--provider", default="mock", show_default=True,
              type=click.Choice(["mock", "openai", "anthropic"]))
@click.option("--model", default=None, help="Model name override")
@click.option("--bandit-path", default=None, help="Path to trained bandit JSON")
@click.option("--strategy", default=None, help="Override strategy for all scenarios")
@click.option("--save-dir", default="results/eval", show_default=True)
@click.option("--port", default=5001, show_default=True)
@click.option("--verbose", is_flag=True, default=False)
def run(split, runs, provider, model, bandit_path, strategy, save_dir, port, verbose):
    """Run evaluation and produce a KPI report."""
    config = EvalConfig(
        runs_per_scenario=runs,
        split=split,
        strategy_override=strategy,
        bandit_path=bandit_path,
        save_dir=save_dir,
        llm_provider=provider,
        llm_model=model,
        app_port=port,
        verbose=verbose,
    )

    logger.info("=" * 60)
    logger.info(f"SQLi Agent Evaluation — split={split} runs={runs}")
    if bandit_path:
        logger.info(f"Using trained bandit: {bandit_path}")
    else:
        logger.info("No bandit provided — using default strategy")
    logger.info("=" * 60)

    harness = EvalHarness(config)
    report = harness.run()
    report.print_summary()

    saved_path = report.save(save_dir)
    print(f"Report saved to: {saved_path}")
    return saved_path


@cli.command()
@click.argument("before_path", type=click.Path(exists=True))
@click.argument("after_path", type=click.Path(exists=True))
@click.option("--threshold", default=0.05, show_default=True,
              help="Regression detection threshold (0-1)")
def compare(before_path, after_path, threshold):
    """Compare two eval reports for regressions and improvements."""
    print_comparison(before_path, after_path)


@cli.command()
@click.option("--provider", default="mock", show_default=True,
              type=click.Choice(["mock", "openai", "anthropic"]))
@click.option("--model", default=None)
@click.option("--save-dir", default="results/eval", show_default=True)
@click.option("--port", default=5001, show_default=True)
def full_pipeline(provider, model, save_dir, port):
    """
    Run the complete before/after pipeline:
    1. Eval with default strategy (pre-training baseline)
    2. Train the bandit
    3. Eval with trained bandit
    4. Compare and show improvement
    """
    from src.rl.trainer import SQLiBanditTrainer, TrainingConfig

    logger.info("Step 1/4: Pre-training evaluation (baseline)")
    pre_config = EvalConfig(
        runs_per_scenario=2,
        split="eval",
        save_dir=save_dir,
        llm_provider=provider,
        llm_model=model,
        app_port=port,
    )
    pre_harness = EvalHarness(pre_config)
    pre_report = pre_harness.run()
    pre_path = pre_report.save(save_dir)
    print(f"Pre-training AFS: {pre_report.afs:.4f}")

    logger.info("Step 2/4: Training (30 episodes)")
    train_config = TrainingConfig(
        num_episodes=30,
        save_dir="results/training",
        llm_provider=provider,
        llm_model=model,
        app_port=port,
    )
    trainer = SQLiBanditTrainer(train_config)
    trainer.train()
    bandit_path = "results/training/bandit_final.json"

    logger.info("Step 3/4: Post-training evaluation")
    post_config = EvalConfig(
        runs_per_scenario=2,
        split="eval",
        bandit_path=bandit_path,
        save_dir=save_dir,
        llm_provider=provider,
        llm_model=model,
        app_port=port,
    )
    post_harness = EvalHarness(post_config)
    post_report = post_harness.run()
    post_path = post_report.save(save_dir)
    print(f"Post-training AFS: {post_report.afs:.4f}")

    logger.info("Step 4/4: Regression comparison")
    print_comparison(pre_path, post_path)
    print(f"AFS improvement: {pre_report.afs:.4f} → {post_report.afs:.4f} "
          f"({post_report.afs - pre_report.afs:+.4f})")


if __name__ == "__main__":
    cli()
