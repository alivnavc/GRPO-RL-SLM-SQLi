"""
Training script: run the UCB Bandit RL training loop.

Usage:
    python scripts/train.py
    python scripts/train.py --episodes 100 --provider openai
    python scripts/train.py --episodes 50 --scenarios train_easy_single train_medium_two
"""

import logging
import os
import sys

import click

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rl.trainer import SQLiBanditTrainer, TrainingConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train")


@click.command()
@click.option("--episodes", default=30, show_default=True, help="Number of training episodes")
@click.option("--provider", default="mock", show_default=True,
              type=click.Choice(["mock", "openai", "anthropic"]),
              help="LLM provider (mock = no API key needed)")
@click.option("--model", default=None, help="Model name override")
@click.option("--save-dir", default="results/training", show_default=True)
@click.option("--port", default=5001, show_default=True, help="Flask app port")
@click.option("--scenarios", multiple=True, default=None, help="Specific scenario IDs to train on")
@click.option("--exploration-c", default=1.41, show_default=True, help="UCB exploration constant")
@click.option("--verbose", is_flag=True, default=False, help="Log every step")
@click.option("--checkpoint-every", default=10, show_default=True)
def main(episodes, provider, model, save_dir, port, scenarios, exploration_c, verbose, checkpoint_every):
    """Train the SQLi agent using UCB Bandit over strategy variants."""

    config = TrainingConfig(
        num_episodes=episodes,
        scenarios=list(scenarios) if scenarios else None,
        exploration_constant=exploration_c,
        save_dir=save_dir,
        checkpoint_every=checkpoint_every,
        verbose=verbose,
        llm_provider=provider,
        llm_model=model,
        app_port=port,
    )

    logger.info("=" * 60)
    logger.info("SQLi Agent RL Training — UCB Bandit")
    logger.info("=" * 60)
    logger.info(f"Provider: {provider} | Episodes: {episodes}")

    trainer = SQLiBanditTrainer(config)
    run = trainer.train()

    summary = next((e for e in reversed(run.episode_logs) if e.get("type") == "summary"), None)
    if summary:
        impr = summary.get("improvement", {})
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        print(f"Pre-training mean score:  {impr.get('mean_score_pre', 0):.4f}")
        print(f"Post-training mean score: {impr.get('mean_score_post', 0):.4f}")
        print(f"Mean improvement:         {impr.get('mean_improvement', 0):+.4f}")
        print()
        print("Per-scenario improvement:")
        for sc_id, sc_data in impr.get("per_scenario", {}).items():
            pct = sc_data.get("relative_improvement_pct", 0)
            delta = sc_data.get("delta", 0)
            print(f"  {sc_id}: {sc_data['pre']:.4f} → {sc_data['post']:.4f}  ({delta:+.4f} / {pct:+.1f}%)")
        print()

    for sc_id in (trainer._train_scenarios or []):
        stats = trainer.bandit.get_arm_stats(sc_id.scenario_id)
        convergence = trainer.bandit.convergence_summary(sc_id.scenario_id)
        print(f"Bandit state for [{sc_id.scenario_id}]:")
        for arm, s in stats.items():
            marker = "★" if arm == convergence["best_arm"] else " "
            print(f"  {marker} {arm}: pulls={s['pulls']} mean={s['mean_reward']:.4f}")
        print(f"  Converged: {convergence['converged']} (best={convergence['best_arm']})")
        print()

    bandit_path = os.path.join(save_dir, "bandit_final.json")
    print(f"Bandit saved to: {bandit_path}")
    print(f"Use this in evaluation with: --bandit-path {bandit_path}")


if __name__ == "__main__":
    main()
