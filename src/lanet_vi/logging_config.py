"""Centralized logging configuration for LaNet-vi."""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    quiet: bool = False,
) -> None:
    """
    Configure logging for LaNet-vi.

    Parameters
    ----------
    level : int
        Logging level (logging.DEBUG, INFO, WARNING, ERROR)
    log_file : Optional[Path]
        If provided, also log to this file
    quiet : bool
        If True, suppress console output (only log to file if provided)

    Examples
    --------
    >>> setup_logging(level=logging.DEBUG)
    >>> setup_logging(level=logging.INFO, log_file=Path("lanet.log"))
    >>> setup_logging(level=logging.WARNING, quiet=True)
    """
    # Root logger configuration
    root_logger = logging.getLogger("lanet_vi")
    root_logger.setLevel(level)
    root_logger.handlers.clear()

    # Formatter with timestamp, logger name, level, and message
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler (unless quiet mode)
    if not quiet:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a specific module.

    Parameters
    ----------
    name : str
        Module name (typically __name__)

    Returns
    -------
    logging.Logger
        Configured logger instance

    Examples
    --------
    >>> logger = get_logger(__name__)
    >>> logger.info("Processing started")
    """
    return logging.getLogger(f"lanet_vi.{name}")
