"""
utils/logger.py
───────────────
Structured JSON logging for production environments.

Every log line is a JSON object, making it trivially parseable by
log-aggregation systems (Datadog, Splunk, CloudWatch, etc.).

Usage:
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Model trained", extra={"rmse": 12345.0, "model": "xgboost"})
"""

import logging
import sys
from pythonjsonlogger import jsonlogger


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Return a module-level logger that emits structured JSON lines.

    Args:
        name:  Usually ``__name__`` so the log line carries the module path.
        level: Logging level (default INFO).

    Returns:
        Configured :class:`logging.Logger` instance.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers when get_logger is called multiple times
    # for the same module name (e.g., during hot-reload in development).
    if logger.handlers:
        return logger

    logger.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    # Include timestamp, log level, logger name, and the message,
    # plus any extra fields passed via ``extra={}``.
    formatter = jsonlogger.JsonFormatter(
        fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Prevent propagation to the root logger to avoid double-printing.
    logger.propagate = False

    return logger
