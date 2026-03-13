"""
Demo script: shows the full agent lifecycle in a self-contained run.
No API key needed — uses the mock LLM backend.

Usage:
    python scripts/demo.py
    python scripts/demo.py --provider openai --episodes 20
"""

import logging
import os
import sys
import time

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import track

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.WARNING)
console = Console()


@click.command()
@click.option("--provider", default="mock", show_default=True,
              type=click.Choice(["mock", "openai", "anthropic"]))
@click.option("--model", default=None, help="Override LLM model")
@click.option("--episodes", default=15, show_default=True, help="Training episodes")
@click.option("--port", default=5001, show_default=True)
def main(provider, model, episodes, port):
    """
    Full demo: start server → pre-eval → train → post-eval → comparison.
    Uses the mock LLM by default (no API key needed).
    """
    from src.agent.react_agent import ReActAgent, LLMBackend
    from src.environment.sqli_env import SQLiEnvironment
    from src.environment.scenarios import get_scenario, SCENARIOS
    from src.eval.harness import EvalConfig, EvalHarness
    from src.rl.trainer import SQLiBanditTrainer, TrainingConfig
    from src.reward.reward_function import score_trajectory

    console.print(Panel.fit(
        "[bold cyan]SQLi Agent RL Demo[/bold cyan]\n"
        "Agent Performance Engineering for Web App Pen Testing\n"
        f"Provider: [green]{provider}[/green]  |  Episodes: [green]{episodes}[/green]",
        border_style="cyan"
    ))

    console.print("\n[bold]Phase 1:[/bold] Starting vulnerable target application...")
    env = SQLiEnvironment(port=port)
    env.start_server()
    console.print(f"  ✅ Flask target running at http://127.0.0.1:{port}")

    console.print("\n[bold]Phase 2:[/bold] Single agent episode walkthrough (systematic_methodical)...")
    llm = LLMBackend(provider=provider, model=model)
    agent = ReActAgent(llm=llm, strategy_id="systematic_methodical")
    scenario = get_scenario("train_medium_two")

    state = env.reset(scenario)
    step_table = Table(title=f"Episode: {scenario.name}", show_lines=True)
    step_table.add_column("Step", style="cyan", width=5)
    step_table.add_column("Tool", style="yellow", width=20)
    step_table.add_column("Signals / Result", style="green")
    step_table.add_column("Reward", style="magenta", width=8)

    total_reward = 0.0
    while not state.episode_done:
        parsed, tokens = agent.act(state, scenario)
        next_state, reward, done, info = env.step(parsed.to_env_action(), tokens_used=tokens)
        total_reward += reward

        signals = info.get("result", {}).get("injection_signals", [])
        findings = len(next_state.findings)
        result_preview = ", ".join(signals) if signals else f"findings={findings}"
        step_table.add_row(
            str(state.step),
            parsed.tool,
            result_preview,
            f"{reward:+.2f}",
        )
        state = next_state

    console.print(step_table)
    metrics = env.get_episode_metrics()
    metrics["total_reward"] = round(total_reward, 4)

    kpi_table = Table(title="Episode KPIs", show_header=True)
    kpi_table.add_column("KPI", style="cyan")
    kpi_table.add_column("Value", style="green")
    kpi_table.add_column("Target", style="yellow")
    kpi_table.add_column("Status", style="bold")

    kpi_rows = [
        ("DCS (Discovery Coverage)", f"{metrics.get('dcs', 0):.3f}", "≥ 0.75",
         "✅" if metrics.get("dcs", 0) >= 0.75 else "❌"),
        ("STC Score (Efficiency)", f"{metrics.get('stc_score', 0):.3f}", "≥ 0.60",
         "✅" if metrics.get("stc_score", 0) >= 0.60 else "❌"),
        ("AVDS (Attack Diversity)", f"{metrics.get('avds', 0):.3f}", "≥ 0.50",
         "✅" if metrics.get("avds", 0) >= 0.50 else "❌"),
        ("TIE (Tool Efficiency)", f"{metrics.get('tie', 0):.3f}", "≥ 0.60",
         "✅" if metrics.get("tie", 0) >= 0.60 else "❌"),
        ("True Positives", str(metrics.get("true_positives", 0)), f"/{metrics.get('known_vulns', 0)}", ""),
        ("False Positives", str(metrics.get("false_positives", 0)), "= 0",
         "✅" if metrics.get("false_positives", 0) == 0 else "❌"),
        ("Total Reward", f"{metrics.get('total_reward', 0):.3f}", "", ""),
        ("Termination", metrics.get("termination_reason", ""), "", ""),
    ]
    for row in kpi_rows:
        kpi_table.add_row(*row)
    console.print(kpi_table)

    console.print(f"\n[bold]Phase 3:[/bold] Pre-training evaluation (default strategy)...")
    pre_config = EvalConfig(
        runs_per_scenario=1,
        split="eval",
        save_dir="results/demo",
        llm_provider=provider,
        llm_model=model,
        app_port=port,
    )
    pre_harness = EvalHarness(pre_config)
    pre_report = pre_harness.run()
    pre_path = pre_report.save("results/demo")
    console.print(f"  Pre-training AFS: [bold red]{pre_report.afs:.4f}[/bold red]")

    console.print(f"\n[bold]Phase 4:[/bold] RL Training ({episodes} episodes, UCB Bandit)...")
    train_config = TrainingConfig(
        num_episodes=episodes,
        save_dir="results/demo/training",
        llm_provider=provider,
        llm_model=model,
        app_port=port,
        verbose=False,
    )
    trainer = SQLiBanditTrainer(train_config)

    with console.status("[cyan]Training in progress...[/cyan]"):
        run = trainer.train()

    summary = next((e for e in reversed(run.episode_logs) if e.get("type") == "summary"), None)

    bandit_table = Table(title="Bandit Arm Statistics (train_medium_two)", show_header=True)
    bandit_table.add_column("Strategy", style="cyan")
    bandit_table.add_column("Pulls", style="yellow")
    bandit_table.add_column("Mean Reward", style="green")
    bandit_table.add_column("UCB Score")

    train_sc_id = "train_medium_two"
    arm_stats = trainer.bandit.get_arm_stats(train_sc_id)
    convergence = trainer.bandit.convergence_summary(train_sc_id)
    for arm, s in sorted(arm_stats.items(), key=lambda x: x[1]["mean_reward"], reverse=True):
        marker = "[bold green]★[/bold green]" if arm == convergence["best_arm"] else " "
        bandit_table.add_row(
            f"{marker} {arm}",
            str(s["pulls"]),
            f"{s['mean_reward']:.4f}",
            str(s["ucb_score"]),
        )
    console.print(bandit_table)
    console.print(
        f"  Bandit converged: [bold]{'YES' if convergence['converged'] else 'NO (need more episodes)'}[/bold]  "
        f"Best arm: [green]{convergence['best_arm']}[/green]"
    )

    console.print(f"\n[bold]Phase 5:[/bold] Post-training evaluation (best bandit arm)...")
    bandit_path = "results/demo/training/bandit_final.json"
    post_config = EvalConfig(
        runs_per_scenario=1,
        split="eval",
        bandit_path=bandit_path,
        save_dir="results/demo",
        llm_provider=provider,
        llm_model=model,
        app_port=port,
    )
    post_harness = EvalHarness(post_config)
    post_report = post_harness.run()
    post_path = post_report.save("results/demo")
    console.print(f"  Post-training AFS: [bold green]{post_report.afs:.4f}[/bold green]")

    console.print(f"\n[bold]Phase 6:[/bold] Before/After comparison...")
    from src.eval.harness import compare_eval_reports

    regression_report = compare_eval_reports(pre_path, post_path)
    afs_delta = post_report.afs - pre_report.afs

    comparison_table = Table(title="Before vs After RL Training", show_header=True)
    comparison_table.add_column("Metric", style="cyan")
    comparison_table.add_column("Before", style="red")
    comparison_table.add_column("After", style="green")
    comparison_table.add_column("Delta", style="bold")
    comparison_table.add_column("Status")

    pre_agg = pre_report.overall_aggregate
    post_agg = post_report.overall_aggregate

    comparison_rows = [
        ("DCS (mean)", pre_agg.get("dcs", {}).get("mean", 0), post_agg.get("dcs", {}).get("mean", 0)),
        ("Composite Score (mean)", pre_agg.get("composite_score", {}).get("mean", 0),
         post_agg.get("composite_score", {}).get("mean", 0)),
        ("STC Score (mean)", pre_agg.get("stc_score", {}).get("mean", 0),
         post_agg.get("stc_score", {}).get("mean", 0)),
        ("AVDS (mean)", pre_agg.get("avds", {}).get("mean", 0),
         post_agg.get("avds", {}).get("mean", 0)),
        ("AFS", pre_report.afs, post_report.afs),
    ]

    for label, before_val, after_val in comparison_rows:
        delta = after_val - before_val
        status = "✅ improved" if delta > 0.005 else ("➡ stable" if abs(delta) <= 0.005 else "❌ regressed")
        comparison_table.add_row(
            label, f"{before_val:.4f}", f"{after_val:.4f}", f"{delta:+.4f}", status
        )

    console.print(comparison_table)

    console.print(Panel.fit(
        f"[bold]Demo Complete[/bold]\n\n"
        f"Pre-training AFS:   [red]{pre_report.afs:.4f}[/red]\n"
        f"Post-training AFS:  [green]{post_report.afs:.4f}[/green]\n"
        f"Improvement:        [bold]{'[green]+' if afs_delta >= 0 else '[red]'}{afs_delta:.4f}[/bold]\n\n"
        f"Regression detected: [bold]{'[red]YES' if regression_report.has_regression else '[green]NO'}[/bold]\n\n"
        f"Results saved to: results/demo/",
        border_style="green" if afs_delta >= 0 else "red"
    ))


if __name__ == "__main__":
    main()
