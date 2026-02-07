"""
Utility modules for OpenUnlearning.
"""

from utils.logging import setup_logging, get_logger

# Lazy imports for optional visualization dependencies
def get_visualization_module():
    """Get visualization module (requires matplotlib)."""
    from utils import visualization
    return visualization

def get_metrics_analysis_module():
    """Get metrics analysis module."""
    from utils import metrics_analysis
    return metrics_analysis

__all__ = [
    "setup_logging",
    "get_logger",
    "get_visualization_module",
    "get_metrics_analysis_module",
]

