"""
Centralized logging configuration for OpenUnlearning.
"""
import logging
import sys
from pathlib import Path
from typing import Optional

import colorlog


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
) -> None:
    """
    Set up centralized logging configuration with color support.
    
    Args:
        level: Logging level (default: INFO)
        log_file: Optional path to log file. If None, logs only to console.
        format_string: Optional custom format string. If None, uses default format.
                      Note: format_string is only used for file logs, console always uses colored format.
    """
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    console_format_string = (
        "%(asctime)s - %(log_color)s%(levelname)s%(reset)s - %(name)s - %(message)s"
    )
    console_formatter = colorlog.ColoredFormatter(
        console_format_string,
        datefmt="%Y-%m-%d %H:%M:%S",
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        },
        secondary_log_colors={},
        style='%'
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if log_file is provided) - always plain text, no colors
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        
        # File format (plain text, no colors)
        if format_string is None:
            format_string = (
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
        file_formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    
    # Set levels for third-party libraries to reduce noise
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("accelerate").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the given name.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)

