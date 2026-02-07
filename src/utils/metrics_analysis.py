"""
Metrics analysis module for unlearning experiments.

This module provides functions to:
- Aggregate and analyze evaluation metrics
- Calculate derived metrics (e.g., forget quality, privacy-utility tradeoff)
- Generate metrics reports and comparisons
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from utils.logging import get_logger

logger = get_logger(__name__)


# Standard metrics for different benchmarks
TOFU_METRICS = [
    "forget_Q_A_ROUGE",
    "forget_Q_A_Prob",
    "model_utility",
    "extraction_strength",
    "privleak",
]

MUSE_METRICS = [
    "forget_ROUGE",
    "forget_Prob",
    "retain_ROUGE",
    "retain_Prob",
    "model_utility",
    "privleak",
]

WMDP_METRICS = [
    "wmdp_accuracy",
    "mmlu_accuracy",
]


def load_metrics_from_summary(path: Union[str, Path]) -> Dict:
    """
    Load evaluation metrics from a summary JSON file.
    
    Args:
        path: Path to directory or summary file
        
    Returns:
        Dictionary with metrics
    """
    path = Path(path)
    
    if path.is_dir():
        # Try to find summary file
        candidates = [
            path / "evals" / "TOFU_SUMMARY.json",
            path / "evals" / "MUSE_SUMMARY.json",
            path / "TOFU_SUMMARY.json",
            path / "MUSE_SUMMARY.json",
        ]
        
        for candidate in candidates:
            if candidate.exists():
                path = candidate
                break
        else:
            # Try any summary file
            summary_files = list(path.glob("**/*_SUMMARY.json"))
            if summary_files:
                path = summary_files[0]
            else:
                raise FileNotFoundError(f"No summary file found in {path}")
    
    with open(path, "r") as f:
        metrics = json.load(f)
    
    return metrics


def load_detailed_metrics(path: Union[str, Path]) -> Dict:
    """
    Load detailed evaluation metrics from an EVAL JSON file.
    
    Args:
        path: Path to directory or EVAL file
        
    Returns:
        Dictionary with detailed metrics
    """
    path = Path(path)
    
    if path.is_dir():
        # Try to find eval file
        candidates = [
            path / "evals" / "TOFU_EVAL.json",
            path / "evals" / "MUSE_EVAL.json",
        ]
        
        for candidate in candidates:
            if candidate.exists():
                path = candidate
                break
        else:
            eval_files = list(path.glob("**/*_EVAL.json"))
            if eval_files:
                path = eval_files[0]
            else:
                raise FileNotFoundError(f"No eval file found in {path}")
    
    with open(path, "r") as f:
        metrics = json.load(f)
    
    return metrics


def calculate_forget_quality(metrics: Dict) -> float:
    """
    Calculate forget quality score.
    
    Higher is better - indicates successful forgetting.
    
    Args:
        metrics: Dictionary with evaluation metrics
        
    Returns:
        Forget quality score (0-1)
    """
    # TOFU-style forget quality
    if "forget_Q_A_ROUGE" in metrics and "forget_Q_A_Prob" in metrics:
        rouge = metrics["forget_Q_A_ROUGE"]
        prob = metrics["forget_Q_A_Prob"]
        # Lower ROUGE and Prob on forget set = better forgetting
        # Normalize to 0-1 where higher is better
        return 1.0 - (rouge + prob) / 2.0
    
    return None


def calculate_privacy_utility_tradeoff(metrics: Dict) -> Dict:
    """
    Calculate privacy-utility tradeoff metrics.
    
    Args:
        metrics: Dictionary with evaluation metrics
        
    Returns:
        Dictionary with tradeoff analysis
    """
    result = {}
    
    utility = metrics.get("model_utility", 0)
    privleak = metrics.get("privleak", 0)
    forget_quality = calculate_forget_quality(metrics)
    
    result["model_utility"] = utility
    result["privacy_leakage"] = privleak
    result["forget_quality"] = forget_quality
    
    # Calculate combined score (higher is better)
    if forget_quality is not None:
        # Weight utility and forget quality, penalize privacy leakage
        result["combined_score"] = (utility + forget_quality) / 2.0 - (privleak / 100.0)
    
    return result


def compare_to_baseline(
    experiment_metrics: Dict,
    baseline_metrics: Dict,
    metric_names: Optional[List[str]] = None,
) -> Dict:
    """
    Compare experiment metrics to baseline.
    
    Args:
        experiment_metrics: Metrics from the experiment
        baseline_metrics: Metrics from the baseline (e.g., fine-tuned model)
        metric_names: List of metrics to compare
        
    Returns:
        Dictionary with comparison results
    """
    if metric_names is None:
        metric_names = list(set(experiment_metrics.keys()) & set(baseline_metrics.keys()))
    
    comparison = {}
    
    for name in metric_names:
        exp_val = experiment_metrics.get(name)
        base_val = baseline_metrics.get(name)
        
        if exp_val is not None and base_val is not None:
            diff = exp_val - base_val
            pct_change = (diff / base_val * 100) if base_val != 0 else 0
            
            comparison[name] = {
                "experiment": exp_val,
                "baseline": base_val,
                "difference": diff,
                "percent_change": pct_change,
            }
    
    return comparison


def rank_experiments(
    experiments: Dict[str, Dict],
    ranking_metrics: Optional[List[str]] = None,
    weights: Optional[Dict[str, float]] = None,
    higher_is_better: Optional[Dict[str, bool]] = None,
) -> List[Tuple[str, float]]:
    """
    Rank experiments based on metrics.
    
    Args:
        experiments: Dict mapping experiment names to their metrics
        ranking_metrics: List of metrics to use for ranking
        weights: Weights for each metric (default: equal weights)
        higher_is_better: Whether higher values are better for each metric
        
    Returns:
        List of (experiment_name, score) tuples, sorted by score (descending)
    """
    if ranking_metrics is None:
        # Use common metrics
        all_metrics = set()
        for metrics in experiments.values():
            all_metrics.update(metrics.keys())
        ranking_metrics = list(all_metrics)
    
    if weights is None:
        weights = {m: 1.0 for m in ranking_metrics}
    
    if higher_is_better is None:
        # Default assumptions
        higher_is_better = {
            "model_utility": True,
            "forget_quality": True,
            "forget_Q_A_ROUGE": False,  # Lower is better (more forgetting)
            "forget_Q_A_Prob": False,
            "extraction_strength": False,
            "privleak": False,
        }
    
    scores = []
    
    for exp_name, metrics in experiments.items():
        score = 0.0
        for metric in ranking_metrics:
            if metric in metrics and metrics[metric] is not None:
                value = metrics[metric]
                weight = weights.get(metric, 1.0)
                
                # Normalize direction
                if not higher_is_better.get(metric, True):
                    value = -value
                
                score += value * weight
        
        scores.append((exp_name, score))
    
    # Sort by score (descending)
    scores.sort(key=lambda x: x[1], reverse=True)
    
    return scores


def calculate_aggregate_statistics(
    experiments: Dict[str, Dict],
) -> Dict[str, Dict]:
    """
    Calculate aggregate statistics across experiments.
    
    Args:
        experiments: Dict mapping experiment names to their metrics
        
    Returns:
        Dictionary with statistics for each metric
    """
    # Collect all values for each metric
    metric_values = {}
    
    for metrics in experiments.values():
        for name, value in metrics.items():
            if value is not None and isinstance(value, (int, float)):
                if name not in metric_values:
                    metric_values[name] = []
                metric_values[name].append(value)
    
    # Calculate statistics
    stats = {}
    for name, values in metric_values.items():
        if values:
            arr = np.array(values)
            stats[name] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "median": float(np.median(arr)),
                "count": len(values),
            }
    
    return stats


def generate_metrics_report(
    experiments: Dict[str, Union[str, Path, Dict]],
    baseline_path: Optional[Union[str, Path]] = None,
    output_path: Optional[Union[str, Path]] = None,
) -> Dict:
    """
    Generate comprehensive metrics report.
    
    Args:
        experiments: Dict mapping experiment names to paths or metrics
        baseline_path: Optional path to baseline for comparison
        output_path: Optional path to save the report
        
    Returns:
        Dictionary with the full report
    """
    # Load all metrics
    all_metrics = {}
    for name, exp in experiments.items():
        if isinstance(exp, (str, Path)):
            try:
                metrics = load_metrics_from_summary(exp)
                all_metrics[name] = metrics
            except Exception as e:
                logger.warning(f"Could not load metrics for {name}: {e}")
        else:
            all_metrics[name] = exp
    
    report = {
        "experiments": all_metrics,
        "aggregate_statistics": calculate_aggregate_statistics(all_metrics),
        "rankings": {},
    }
    
    # Add rankings
    ranking = rank_experiments(all_metrics)
    report["rankings"]["combined"] = ranking
    
    # Add privacy-utility analysis
    report["privacy_utility"] = {}
    for name, metrics in all_metrics.items():
        report["privacy_utility"][name] = calculate_privacy_utility_tradeoff(metrics)
    
    # Compare to baseline if provided
    if baseline_path is not None:
        try:
            baseline_metrics = load_metrics_from_summary(baseline_path)
            report["baseline"] = baseline_metrics
            report["comparisons"] = {}
            
            for name, metrics in all_metrics.items():
                report["comparisons"][name] = compare_to_baseline(metrics, baseline_metrics)
        except Exception as e:
            logger.warning(f"Could not load baseline metrics: {e}")
    
    # Save report if path provided
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Metrics report saved to: {output_path}")
    
    return report


def print_metrics_table(experiments: Dict[str, Dict], metrics: Optional[List[str]] = None):
    """
    Print a formatted table of metrics.
    
    Args:
        experiments: Dict mapping experiment names to their metrics
        metrics: List of metrics to display (None for all)
    """
    if not experiments:
        print("No experiments to display")
        return
    
    if metrics is None:
        metrics = set()
        for exp_metrics in experiments.values():
            metrics.update(exp_metrics.keys())
        metrics = sorted(list(metrics))
    
    # Calculate column widths
    name_width = max(len(name) for name in experiments.keys())
    name_width = max(name_width, 15)
    
    # Print header
    header = f"{'Experiment':<{name_width}}"
    for m in metrics:
        header += f" | {m:>15}"
    print(header)
    print("-" * len(header))
    
    # Print rows
    for name, exp_metrics in experiments.items():
        row = f"{name:<{name_width}}"
        for m in metrics:
            value = exp_metrics.get(m)
            if value is None:
                row += f" | {'N/A':>15}"
            elif isinstance(value, float):
                row += f" | {value:>15.4f}"
            else:
                row += f" | {str(value):>15}"
        print(row)
    
    print()

