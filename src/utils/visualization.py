"""
Visualization module for training/unlearning loss curves and metrics.

This module provides functions to:
- Plot loss curves from trainer_state.json files
- Compare multiple experiments (fine-tuning vs unlearning methods)
- Visualize evaluation metrics from SUMMARY.json files
- Generate comprehensive dashboards
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

from utils.logging import get_logger

logger = get_logger(__name__)


# Color palette for different experiments/methods
METHOD_COLORS = {
    # Fine-tuning
    "finetune": "#2ecc71",
    "full": "#27ae60",
    "retain": "#1abc9c",
    # Unlearning methods
    "GradAscent": "#e74c3c",
    "GradDiff": "#9b59b6",
    "SimNPO": "#3498db",
    "NPO": "#2980b9",
    "DPO": "#f39c12",
    "RMU": "#e67e22",
    "CEU": "#d35400",
    "PDU": "#c0392b",
    "WGA": "#8e44ad",
    "UNDIAL": "#16a085",
    "SatImp": "#1abc9c",
}

# Default style settings
STYLE_CONFIG = {
    "figure.figsize": (12, 8),
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "lines.linewidth": 2,
    "lines.markersize": 6,
    "grid.alpha": 0.3,
}


def check_dependencies():
    """Check if required visualization dependencies are available."""
    if not HAS_MATPLOTLIB:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install it with: pip install matplotlib"
        )


def setup_style():
    """Set up matplotlib style for consistent visualizations."""
    check_dependencies()
    
    plt.rcParams.update(STYLE_CONFIG)
    
    if HAS_SEABORN:
        sns.set_theme(style="whitegrid", palette="deep")
    else:
        plt.style.use("seaborn-v0_8-whitegrid")


def load_trainer_state(path: Union[str, Path]) -> Dict:
    """
    Load trainer_state.json from a directory or file path.
    
    Args:
        path: Path to trainer_state.json or directory containing it
        
    Returns:
        Dictionary with trainer state data
    """
    path = Path(path)
    
    if path.is_dir():
        state_file = path / "trainer_state.json"
    else:
        state_file = path
        
    if not state_file.exists():
        raise FileNotFoundError(f"Trainer state file not found: {state_file}")
        
    with open(state_file, "r") as f:
        data = json.load(f)
        
    logger.debug(f"Loaded trainer state from: {state_file}")
    return data


def load_eval_summary(path: Union[str, Path], suffix: str = "SUMMARY") -> Dict:
    """
    Load evaluation summary JSON file.
    
    Args:
        path: Path to directory containing eval summary or direct file path
        suffix: Suffix of the summary file (default: SUMMARY)
        
    Returns:
        Dictionary with evaluation metrics
    """
    path = Path(path)
    
    if path.is_dir():
        # Try to find summary file
        summary_files = list(path.glob(f"*_{suffix}.json"))
        if not summary_files:
            # Try in evals subdirectory
            evals_dir = path / "evals"
            if evals_dir.exists():
                summary_files = list(evals_dir.glob(f"*_{suffix}.json"))
        
        if not summary_files:
            raise FileNotFoundError(f"No summary file found in: {path}")
            
        summary_file = summary_files[0]
    else:
        summary_file = path
        
    if not summary_file.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_file}")
        
    with open(summary_file, "r") as f:
        data = json.load(f)
        
    logger.debug(f"Loaded eval summary from: {summary_file}")
    return data


def extract_training_history(trainer_state: Dict) -> Dict[str, List]:
    """
    Extract training history from trainer state.
    
    Args:
        trainer_state: Loaded trainer state dictionary
        
    Returns:
        Dictionary with lists for each metric (step, epoch, loss, etc.)
    """
    log_history = trainer_state.get("log_history", [])
    
    history = {
        "step": [],
        "epoch": [],
        "loss": [],
        "learning_rate": [],
        "grad_norm": [],
        "eval_metrics": {},
    }
    
    for entry in log_history:
        step = entry.get("step")
        
        # Training logs (with loss)
        if "loss" in entry:
            history["step"].append(step)
            history["epoch"].append(entry.get("epoch", 0))
            history["loss"].append(entry["loss"])
            history["learning_rate"].append(entry.get("learning_rate", 0))
            history["grad_norm"].append(entry.get("grad_norm", 0))
            
        # Evaluation logs (without loss, with other metrics)
        elif step is not None and "loss" not in entry and "train_loss" not in entry:
            for key, value in entry.items():
                if key not in ["step", "epoch"] and isinstance(value, (int, float)):
                    if key not in history["eval_metrics"]:
                        history["eval_metrics"][key] = {"steps": [], "values": []}
                    history["eval_metrics"][key]["steps"].append(step)
                    history["eval_metrics"][key]["values"].append(value)
    
    return history


def plot_loss_curve(
    trainer_state: Dict,
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    color: Optional[str] = None,
    label: Optional[str] = None,
    x_axis: str = "step",
    show_smoothed: bool = True,
    smooth_weight: float = 0.9,
) -> plt.Axes:
    """
    Plot loss curve from trainer state.
    
    Args:
        trainer_state: Loaded trainer state dictionary
        title: Plot title
        ax: Matplotlib axes to plot on (creates new if None)
        color: Line color
        label: Line label for legend
        x_axis: X-axis type ('step' or 'epoch')
        show_smoothed: Whether to show smoothed curve
        smooth_weight: Smoothing weight (EMA factor)
        
    Returns:
        Matplotlib axes object
    """
    check_dependencies()
    setup_style()
    
    history = extract_training_history(trainer_state)
    
    if not history["loss"]:
        logger.warning("No loss values found in trainer state")
        return ax
    
    if ax is None:
        fig, ax = plt.subplots(figsize=STYLE_CONFIG["figure.figsize"])
    
    x_data = history[x_axis]
    y_data = history["loss"]
    
    color = color or "#3498db"
    label = label or "Loss"
    
    # Plot raw loss
    ax.plot(x_data, y_data, color=color, alpha=0.3, linewidth=1)
    
    # Plot smoothed loss
    if show_smoothed:
        smoothed = exponential_moving_average(y_data, smooth_weight)
        ax.plot(x_data, smoothed, color=color, label=label, linewidth=2)
    else:
        ax.plot(x_data, y_data, color=color, label=label, linewidth=2)
    
    ax.set_xlabel(x_axis.capitalize())
    ax.set_ylabel("Loss")
    
    if title:
        ax.set_title(title)
        
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return ax


def exponential_moving_average(data: List[float], weight: float = 0.9) -> List[float]:
    """Calculate exponential moving average for smoothing."""
    smoothed = []
    last = data[0] if data else 0
    for value in data:
        smoothed_val = last * weight + value * (1 - weight)
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def compare_experiments(
    experiments: Dict[str, Union[str, Path, Dict]],
    title: str = "Training Loss Comparison",
    x_axis: str = "step",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 8),
    show_grad_norm: bool = True,
    show_lr: bool = True,
) -> plt.Figure:
    """
    Compare loss curves from multiple experiments.
    
    Args:
        experiments: Dict mapping experiment names to paths or trainer states
        title: Plot title
        x_axis: X-axis type ('step' or 'epoch')
        save_path: Path to save the figure
        figsize: Figure size
        show_grad_norm: Whether to show gradient norm subplot
        show_lr: Whether to show learning rate subplot
        
    Returns:
        Matplotlib figure object
    """
    check_dependencies()
    setup_style()
    
    n_subplots = 1 + int(show_grad_norm) + int(show_lr)
    fig, axes = plt.subplots(n_subplots, 1, figsize=figsize, sharex=True)
    
    if n_subplots == 1:
        axes = [axes]
    
    ax_loss = axes[0]
    ax_idx = 1
    ax_grad = axes[ax_idx] if show_grad_norm else None
    ax_idx += int(show_grad_norm)
    ax_lr = axes[ax_idx] if show_lr else None
    
    for exp_name, exp_data in experiments.items():
        # Load trainer state if path is provided
        if isinstance(exp_data, (str, Path)):
            trainer_state = load_trainer_state(exp_data)
        else:
            trainer_state = exp_data
            
        history = extract_training_history(trainer_state)
        
        if not history["loss"]:
            logger.warning(f"No loss values found for experiment: {exp_name}")
            continue
        
        # Determine color based on method name
        color = get_method_color(exp_name)
        
        x_data = history[x_axis]
        
        # Plot loss
        smoothed_loss = exponential_moving_average(history["loss"])
        ax_loss.plot(x_data, history["loss"], color=color, alpha=0.2, linewidth=1)
        ax_loss.plot(x_data, smoothed_loss, color=color, label=exp_name, linewidth=2)
        
        # Plot gradient norm
        if show_grad_norm and ax_grad is not None and history["grad_norm"]:
            smoothed_grad = exponential_moving_average(history["grad_norm"])
            ax_grad.plot(x_data, smoothed_grad, color=color, label=exp_name, linewidth=2)
        
        # Plot learning rate
        if show_lr and ax_lr is not None and history["learning_rate"]:
            ax_lr.plot(x_data, history["learning_rate"], color=color, label=exp_name, linewidth=2)
    
    # Configure axes
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title(title)
    ax_loss.legend(loc="upper right")
    ax_loss.grid(True, alpha=0.3)
    
    if show_grad_norm and ax_grad is not None:
        ax_grad.set_ylabel("Gradient Norm")
        ax_grad.set_title("Gradient Norm During Training")
        ax_grad.legend(loc="upper right")
        ax_grad.grid(True, alpha=0.3)
    
    if show_lr and ax_lr is not None:
        ax_lr.set_ylabel("Learning Rate")
        ax_lr.set_xlabel(x_axis.capitalize())
        ax_lr.set_title("Learning Rate Schedule")
        ax_lr.legend(loc="upper right")
        ax_lr.grid(True, alpha=0.3)
    else:
        axes[-1].set_xlabel(x_axis.capitalize())
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig


def get_method_color(name: str) -> str:
    """Get color for a method based on its name."""
    for method, color in METHOD_COLORS.items():
        if method.lower() in name.lower():
            return color
    # Return a default color from the palette
    hash_val = hash(name) % len(METHOD_COLORS)
    return list(METHOD_COLORS.values())[hash_val]


def plot_metrics_comparison(
    experiments: Dict[str, Union[str, Path, Dict]],
    metrics: Optional[List[str]] = None,
    title: str = "Evaluation Metrics Comparison",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 8),
) -> plt.Figure:
    """
    Plot comparison of evaluation metrics across experiments.
    
    Args:
        experiments: Dict mapping experiment names to paths or summary dicts
        metrics: List of metric names to plot (None for all)
        title: Plot title
        save_path: Path to save the figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure object
    """
    check_dependencies()
    setup_style()
    
    # Load all summaries
    all_metrics = {}
    for exp_name, exp_data in experiments.items():
        if isinstance(exp_data, (str, Path)):
            try:
                summary = load_eval_summary(exp_data)
            except FileNotFoundError:
                logger.warning(f"No eval summary found for: {exp_name}")
                continue
        else:
            summary = exp_data
            
        all_metrics[exp_name] = summary
    
    if not all_metrics:
        logger.error("No metrics found for any experiment")
        return None
    
    # Determine which metrics to plot
    if metrics is None:
        # Get all unique metrics
        all_metric_names = set()
        for summary in all_metrics.values():
            all_metric_names.update(summary.keys())
        metrics = sorted(list(all_metric_names))
    
    if not metrics:
        logger.error("No metrics to plot")
        return None
    
    # Create bar chart
    n_metrics = len(metrics)
    n_experiments = len(all_metrics)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(n_metrics)
    width = 0.8 / n_experiments
    
    for i, (exp_name, summary) in enumerate(all_metrics.items()):
        values = [summary.get(m, 0) or 0 for m in metrics]
        color = get_method_color(exp_name)
        offset = (i - n_experiments / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=exp_name, color=color, alpha=0.8)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            if value != 0:
                height = bar.get_height()
                ax.annotate(
                    f'{value:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=8, rotation=45
                )
    
    ax.set_ylabel("Value")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=45, ha='right')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig


def plot_unlearning_dashboard(
    finetune_path: Union[str, Path],
    unlearn_paths: Dict[str, Union[str, Path]],
    title: str = "Unlearning Dashboard",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 12),
) -> plt.Figure:
    """
    Create a comprehensive dashboard comparing fine-tuning and unlearning.
    
    Args:
        finetune_path: Path to fine-tuned model directory
        unlearn_paths: Dict mapping method names to unlearned model directories
        title: Dashboard title
        save_path: Path to save the figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure object
    """
    check_dependencies()
    setup_style()
    
    fig = plt.figure(figsize=figsize)
    
    # Create grid for subplots
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)
    
    ax_loss = fig.add_subplot(gs[0, 0])
    ax_grad = fig.add_subplot(gs[0, 1])
    ax_metrics = fig.add_subplot(gs[1, 0])
    ax_summary = fig.add_subplot(gs[1, 1])
    
    # Collect all experiments
    experiments = {"Fine-tune": finetune_path}
    experiments.update(unlearn_paths)
    
    # Plot loss curves
    for exp_name, exp_path in experiments.items():
        try:
            trainer_state = load_trainer_state(exp_path)
            history = extract_training_history(trainer_state)
            
            if history["loss"]:
                color = get_method_color(exp_name)
                x_data = history["step"]
                smoothed = exponential_moving_average(history["loss"])
                
                ax_loss.plot(x_data, smoothed, color=color, label=exp_name, linewidth=2)
                
                if history["grad_norm"]:
                    grad_smoothed = exponential_moving_average(history["grad_norm"])
                    ax_grad.plot(x_data, grad_smoothed, color=color, label=exp_name, linewidth=2)
        except Exception as e:
            logger.warning(f"Error loading trainer state for {exp_name}: {e}")
    
    ax_loss.set_xlabel("Step")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Training Loss")
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)
    
    ax_grad.set_xlabel("Step")
    ax_grad.set_ylabel("Gradient Norm")
    ax_grad.set_title("Gradient Norm")
    ax_grad.legend()
    ax_grad.grid(True, alpha=0.3)
    
    # Load and plot metrics
    all_metrics = {}
    for exp_name, exp_path in experiments.items():
        try:
            summary = load_eval_summary(exp_path)
            all_metrics[exp_name] = summary
        except Exception as e:
            logger.warning(f"Error loading eval summary for {exp_name}: {e}")
    
    if all_metrics:
        # Get common metrics
        common_metrics = set.intersection(*[set(m.keys()) for m in all_metrics.values()])
        common_metrics = sorted(list(common_metrics))
        
        if common_metrics:
            x = np.arange(len(common_metrics))
            width = 0.8 / len(all_metrics)
            
            for i, (exp_name, summary) in enumerate(all_metrics.items()):
                values = [summary.get(m, 0) or 0 for m in common_metrics]
                color = get_method_color(exp_name)
                offset = (i - len(all_metrics) / 2 + 0.5) * width
                ax_metrics.bar(x + offset, values, width, label=exp_name, color=color, alpha=0.8)
            
            ax_metrics.set_xticks(x)
            ax_metrics.set_xticklabels(common_metrics, rotation=45, ha='right')
            ax_metrics.set_ylabel("Value")
            ax_metrics.set_title("Evaluation Metrics")
            ax_metrics.legend()
            ax_metrics.grid(True, alpha=0.3, axis='y')
    
    # Summary table
    ax_summary.axis('off')
    
    if all_metrics:
        # Create summary table data
        table_data = []
        headers = ["Method"] + common_metrics
        
        for exp_name, summary in all_metrics.items():
            row = [exp_name] + [f"{summary.get(m, 0):.4f}" for m in common_metrics]
            table_data.append(row)
        
        table = ax_summary.table(
            cellText=table_data,
            colLabels=headers,
            loc='center',
            cellLoc='center',
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        ax_summary.set_title("Metrics Summary", fontsize=14, pad=20)
    
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig


def plot_training_progress(
    path: Union[str, Path],
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 10),
) -> plt.Figure:
    """
    Plot detailed training progress including loss, LR, grad norm, and metrics.
    
    Args:
        path: Path to experiment directory
        title: Plot title
        save_path: Path to save the figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure object
    """
    check_dependencies()
    setup_style()
    
    trainer_state = load_trainer_state(path)
    history = extract_training_history(trainer_state)
    
    n_subplots = 3
    if history["eval_metrics"]:
        n_subplots += 1
    
    fig, axes = plt.subplots(n_subplots, 1, figsize=figsize, sharex=True)
    
    x_data = history["step"]
    
    # Loss
    ax = axes[0]
    if history["loss"]:
        smoothed = exponential_moving_average(history["loss"])
        ax.plot(x_data, history["loss"], alpha=0.3, color="#3498db", linewidth=1)
        ax.plot(x_data, smoothed, color="#3498db", linewidth=2, label="Loss (smoothed)")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Learning Rate
    ax = axes[1]
    if history["learning_rate"]:
        ax.plot(x_data, history["learning_rate"], color="#e74c3c", linewidth=2)
        ax.set_ylabel("Learning Rate")
        ax.set_title("Learning Rate Schedule")
        ax.grid(True, alpha=0.3)
    
    # Gradient Norm
    ax = axes[2]
    if history["grad_norm"]:
        smoothed = exponential_moving_average(history["grad_norm"])
        ax.plot(x_data, history["grad_norm"], alpha=0.3, color="#2ecc71", linewidth=1)
        ax.plot(x_data, smoothed, color="#2ecc71", linewidth=2, label="Grad Norm (smoothed)")
        ax.set_ylabel("Gradient Norm")
        ax.set_title("Gradient Norm")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Eval Metrics (if available)
    if history["eval_metrics"] and n_subplots > 3:
        ax = axes[3]
        for metric_name, metric_data in history["eval_metrics"].items():
            ax.plot(
                metric_data["steps"],
                metric_data["values"],
                marker='o',
                label=metric_name,
                linewidth=2,
            )
        ax.set_ylabel("Metric Value")
        ax.set_title("Evaluation Metrics")
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel("Step")
    
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig


def save_figure(fig: plt.Figure, path: str, dpi: int = 150):
    """Save figure to file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
    logger.info(f"Figure saved to: {path}")


def discover_experiments(
    base_dir: Union[str, Path],
    pattern: str = "*",
) -> Dict[str, Path]:
    """
    Discover experiment directories in a base directory.
    
    Args:
        base_dir: Base directory to search
        pattern: Glob pattern for filtering directories
        
    Returns:
        Dict mapping experiment names to paths
    """
    base_dir = Path(base_dir)
    experiments = {}
    
    for path in base_dir.glob(pattern):
        if path.is_dir():
            # Check if this looks like an experiment directory
            trainer_state = path / "trainer_state.json"
            if trainer_state.exists():
                experiments[path.name] = path
    
    logger.info(f"Discovered {len(experiments)} experiments in {base_dir}")
    return experiments


def calculate_aggregate_metrics(experiments: Dict[str, Union[str, Path]]) -> Dict[str, Dict]:
    """
    Calculate aggregate metrics across experiments.
    
    Args:
        experiments: Dict mapping experiment names to paths
        
    Returns:
        Dict with aggregate statistics for each metric
    """
    all_metrics = {}
    
    for exp_name, exp_path in experiments.items():
        try:
            summary = load_eval_summary(exp_path)
            all_metrics[exp_name] = summary
        except Exception as e:
            logger.warning(f"Error loading metrics for {exp_name}: {e}")
    
    if not all_metrics:
        return {}
    
    # Get all metric names
    metric_names = set()
    for summary in all_metrics.values():
        metric_names.update(summary.keys())
    
    # Calculate statistics
    aggregates = {}
    for metric in metric_names:
        values = [
            summary.get(metric) 
            for summary in all_metrics.values() 
            if summary.get(metric) is not None
        ]
        
        if values:
            aggregates[metric] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
                "count": len(values),
            }
    
    return aggregates


def generate_report(
    experiments: Dict[str, Union[str, Path]],
    output_dir: Union[str, Path],
    title: str = "Training Report",
) -> str:
    """
    Generate a comprehensive visualization report.
    
    Args:
        experiments: Dict mapping experiment names to paths
        output_dir: Directory to save report files
        title: Report title
        
    Returns:
        Path to the main report file
    """
    check_dependencies()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Generating report for {len(experiments)} experiments")
    
    # Generate comparison plots
    compare_fig = compare_experiments(
        experiments,
        title=f"{title} - Loss Comparison",
        save_path=output_dir / "loss_comparison.png",
    )
    plt.close(compare_fig)
    
    # Generate metrics comparison
    metrics_fig = plot_metrics_comparison(
        experiments,
        title=f"{title} - Metrics Comparison",
        save_path=output_dir / "metrics_comparison.png",
    )
    if metrics_fig:
        plt.close(metrics_fig)
    
    # Generate individual training progress plots
    for exp_name, exp_path in experiments.items():
        try:
            fig = plot_training_progress(
                exp_path,
                title=exp_name,
                save_path=output_dir / f"{exp_name}_progress.png",
            )
            plt.close(fig)
        except Exception as e:
            logger.warning(f"Error generating progress plot for {exp_name}: {e}")
    
    # Save aggregate metrics
    aggregates = calculate_aggregate_metrics(experiments)
    if aggregates:
        with open(output_dir / "aggregate_metrics.json", "w") as f:
            json.dump(aggregates, f, indent=2)
    
    logger.info(f"Report generated in: {output_dir}")
    return str(output_dir)

