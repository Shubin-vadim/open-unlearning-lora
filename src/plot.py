"""
CLI interface for plotting training/unlearning loss curves and metrics.

Usage examples:
    # Plot single experiment
    python -m plot --path saves/finetune/tofu_Qwen2.5-3B-Instruct_full

    # Compare multiple experiments
    python -m plot compare \
        --experiments saves/finetune/tofu_Qwen2.5-3B-Instruct_full \
        --experiments saves/unlearn/tofu_Qwen2.5-3B-Instruct_forget10_GradAscent \
        --output plots/comparison.png

    # Generate full report
    python -m plot report \
        --base-dir saves/unlearn \
        --output reports/unlearning_report

    # Create dashboard
    python -m plot dashboard \
        --finetune saves/finetune/tofu_Qwen2.5-3B-Instruct_full \
        --unlearn saves/unlearn/tofu_Qwen2.5-3B-Instruct_forget10_GradAscent \
        --unlearn saves/unlearn/tofu_Qwen2.5-3B-Instruct_forget10_GradDiff \
        --output plots/dashboard.png
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend by default
import matplotlib.pyplot as plt

from utils.logging import setup_logging, get_logger
from utils.visualization import (
    plot_training_progress,
    compare_experiments,
    plot_metrics_comparison,
    plot_unlearning_dashboard,
    discover_experiments,
    generate_report,
    load_trainer_state,
    load_eval_summary,
    check_dependencies,
)

logger = get_logger(__name__)


def cmd_plot_single(args):
    """Plot training progress for a single experiment."""
    check_dependencies()
    
    logger.info(f"Plotting training progress for: {args.path}")
    
    # If no output specified, save to plots/ directory in experiment folder
    if not args.output:
        exp_path = Path(args.path)
        plots_dir = exp_path / "plots"
        plots_dir.mkdir(exist_ok=True)
        args.output = str(plots_dir / "training_progress.png")
        logger.info(f"No output path specified, saving to: {args.output}")
    
    fig = plot_training_progress(
        path=args.path,
        title=args.title or Path(args.path).name,
        save_path=args.output,
    )
    
    if args.show:
        # Switch to interactive backend for showing
        matplotlib.use('TkAgg')  # or 'Qt5Agg' depending on system
        plt.show()
    
    logger.info(f"Plot saved to: {args.output}")
    
    plt.close(fig)


def cmd_compare(args):
    """Compare multiple experiments."""
    check_dependencies()
    
    # Build experiments dict from paths
    experiments = {}
    for path in args.experiments:
        path = Path(path)
        name = args.names.pop(0) if args.names else path.name
        experiments[name] = path
    
    logger.info(f"Comparing {len(experiments)} experiments")
    
    # If no output specified, save to current directory or first experiment's plots folder
    if not args.output:
        first_exp_path = Path(list(experiments.values())[0])
        plots_dir = first_exp_path / "plots"
        plots_dir.mkdir(exist_ok=True)
        args.output = str(plots_dir / "comparison.png")
        logger.info(f"No output path specified, saving to: {args.output}")
    
    fig = compare_experiments(
        experiments=experiments,
        title=args.title or "Training Comparison",
        x_axis=args.x_axis,
        save_path=args.output,
        show_grad_norm=not args.no_grad,
        show_lr=not args.no_lr,
    )
    
    if args.show:
        matplotlib.use('TkAgg')
        plt.show()
    
    logger.info(f"Plot saved to: {args.output}")
    
    plt.close(fig)


def cmd_metrics(args):
    """Compare evaluation metrics."""
    check_dependencies()
    
    # Build experiments dict from paths
    experiments = {}
    for path in args.experiments:
        path = Path(path)
        experiments[path.name] = path
    
    logger.info(f"Comparing metrics for {len(experiments)} experiments")
    
    # If no output specified, save to first experiment's plots folder
    if not args.output:
        first_exp_path = Path(list(experiments.values())[0])
        plots_dir = first_exp_path / "plots"
        plots_dir.mkdir(exist_ok=True)
        args.output = str(plots_dir / "metrics_comparison.png")
        logger.info(f"No output path specified, saving to: {args.output}")
    
    fig = plot_metrics_comparison(
        experiments=experiments,
        metrics=args.metrics if args.metrics else None,
        title=args.title or "Metrics Comparison",
        save_path=args.output,
    )
    
    if fig:
        if args.show:
            matplotlib.use('TkAgg')
            plt.show()
        
        logger.info(f"Plot saved to: {args.output}")
        
        plt.close(fig)
    else:
        logger.error("Failed to generate metrics comparison")


def cmd_dashboard(args):
    """Create unlearning dashboard."""
    check_dependencies()
    
    # Build unlearn dict from paths
    unlearn_paths = {}
    for path in args.unlearn:
        path = Path(path)
        unlearn_paths[path.name] = path
    
    logger.info(f"Creating dashboard with {len(unlearn_paths)} unlearning methods")
    
    # If no output specified, save to finetune experiment's plots folder
    if not args.output:
        finetune_path = Path(args.finetune)
        plots_dir = finetune_path / "plots"
        plots_dir.mkdir(exist_ok=True)
        args.output = str(plots_dir / "dashboard.png")
        logger.info(f"No output path specified, saving to: {args.output}")
    
    fig = plot_unlearning_dashboard(
        finetune_path=args.finetune,
        unlearn_paths=unlearn_paths,
        title=args.title or "Unlearning Dashboard",
        save_path=args.output,
    )
    
    if args.show:
        matplotlib.use('TkAgg')
        plt.show()
    
    logger.info(f"Dashboard saved to: {args.output}")
    
    plt.close(fig)


def cmd_report(args):
    """Generate comprehensive report."""
    check_dependencies()
    
    # Discover experiments
    experiments = discover_experiments(args.base_dir, args.pattern)
    
    if not experiments:
        logger.error(f"No experiments found in {args.base_dir}")
        return
    
    logger.info(f"Found {len(experiments)} experiments")
    
    report_path = generate_report(
        experiments=experiments,
        output_dir=args.output,
        title=args.title or "Training Report",
    )
    
    logger.info(f"Report generated: {report_path}")


def cmd_discover(args):
    """Discover and list experiments."""
    experiments = discover_experiments(args.base_dir, args.pattern)
    
    print(f"\nFound {len(experiments)} experiments in {args.base_dir}:\n")
    for name, path in sorted(experiments.items()):
        print(f"  • {name}")
        print(f"    Path: {path}")
        
        # Try to load some info
        try:
            state = load_trainer_state(path)
            print(f"    Steps: {state.get('global_step', 'N/A')}")
            print(f"    Epochs: {state.get('epoch', 'N/A')}")
        except Exception:
            pass
        
        try:
            summary = load_eval_summary(path)
            metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in list(summary.items())[:3]])
            print(f"    Metrics: {metrics_str}")
        except Exception:
            pass
        
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Visualization tools for training and unlearning experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Single plot command
    single_parser = subparsers.add_parser(
        "single",
        help="Plot training progress for a single experiment",
    )
    single_parser.add_argument(
        "--path", "-p",
        required=True,
        help="Path to experiment directory",
    )
    single_parser.add_argument(
        "--output", "-o",
        help="Output file path (if not set, shows interactive plot)",
    )
    single_parser.add_argument(
        "--title", "-t",
        help="Plot title",
    )
    single_parser.add_argument(
        "--show",
        action="store_true",
        help="Show interactive plot even when saving",
    )
    single_parser.set_defaults(func=cmd_plot_single)
    
    # Compare command
    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare multiple experiments",
    )
    compare_parser.add_argument(
        "--experiments", "-e",
        action="append",
        required=True,
        help="Paths to experiment directories (can be repeated)",
    )
    compare_parser.add_argument(
        "--names", "-n",
        action="append",
        default=[],
        help="Custom names for experiments (can be repeated)",
    )
    compare_parser.add_argument(
        "--output", "-o",
        help="Output file path",
    )
    compare_parser.add_argument(
        "--title", "-t",
        help="Plot title",
    )
    compare_parser.add_argument(
        "--x-axis", "-x",
        choices=["step", "epoch"],
        default="step",
        help="X-axis type (default: step)",
    )
    compare_parser.add_argument(
        "--no-grad",
        action="store_true",
        help="Don't show gradient norm subplot",
    )
    compare_parser.add_argument(
        "--no-lr",
        action="store_true",
        help="Don't show learning rate subplot",
    )
    compare_parser.add_argument(
        "--show",
        action="store_true",
        help="Show interactive plot",
    )
    compare_parser.set_defaults(func=cmd_compare)
    
    # Metrics command
    metrics_parser = subparsers.add_parser(
        "metrics",
        help="Compare evaluation metrics",
    )
    metrics_parser.add_argument(
        "--experiments", "-e",
        action="append",
        required=True,
        help="Paths to experiment directories",
    )
    metrics_parser.add_argument(
        "--metrics", "-m",
        action="append",
        help="Specific metrics to plot (default: all)",
    )
    metrics_parser.add_argument(
        "--output", "-o",
        help="Output file path",
    )
    metrics_parser.add_argument(
        "--title", "-t",
        help="Plot title",
    )
    metrics_parser.add_argument(
        "--show",
        action="store_true",
        help="Show interactive plot",
    )
    metrics_parser.set_defaults(func=cmd_metrics)
    
    # Dashboard command
    dashboard_parser = subparsers.add_parser(
        "dashboard",
        help="Create unlearning dashboard",
    )
    dashboard_parser.add_argument(
        "--finetune", "-f",
        required=True,
        help="Path to fine-tuned model directory",
    )
    dashboard_parser.add_argument(
        "--unlearn", "-u",
        action="append",
        required=True,
        help="Paths to unlearned model directories",
    )
    dashboard_parser.add_argument(
        "--output", "-o",
        help="Output file path",
    )
    dashboard_parser.add_argument(
        "--title", "-t",
        help="Dashboard title",
    )
    dashboard_parser.add_argument(
        "--show",
        action="store_true",
        help="Show interactive plot",
    )
    dashboard_parser.set_defaults(func=cmd_dashboard)
    
    # Report command
    report_parser = subparsers.add_parser(
        "report",
        help="Generate comprehensive report",
    )
    report_parser.add_argument(
        "--base-dir", "-d",
        required=True,
        help="Base directory containing experiments",
    )
    report_parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output directory for report",
    )
    report_parser.add_argument(
        "--pattern",
        default="*",
        help="Glob pattern for filtering directories",
    )
    report_parser.add_argument(
        "--title", "-t",
        help="Report title",
    )
    report_parser.set_defaults(func=cmd_report)
    
    # Discover command
    discover_parser = subparsers.add_parser(
        "discover",
        help="Discover and list experiments",
    )
    discover_parser.add_argument(
        "--base-dir", "-d",
        required=True,
        help="Base directory to search",
    )
    discover_parser.add_argument(
        "--pattern",
        default="*",
        help="Glob pattern for filtering",
    )
    discover_parser.set_defaults(func=cmd_discover)
    
    # Parse and execute
    args = parser.parse_args()
    
    setup_logging()
    
    if args.command is None:
        parser.print_help()
        return
    
    args.func(args)


if __name__ == "__main__":
    main()

