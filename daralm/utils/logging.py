"""Structured logging setup shared across scripts and modules.

Every script (``inspect_model_config.py``, and later ``train.py``,
``generate.py``, ...) should log through a logger from this module rather
than ad hoc ``print()`` calls, so log output stays consistent (timestamp,
level, module name) as the project grows.
"""

from __future__ import annotations

import logging

_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Return a configured logger for ``name``.

    Safe to call repeatedly for the same name (e.g. across module reloads
    or repeated imports) — it will not attach duplicate handlers, so log
    lines won't be printed multiple times.

    Args:
        name: Usually ``__name__`` of the calling module.
        level: Logging level name, e.g. "DEBUG", "INFO", "WARNING", "ERROR".
    """
    logger = logging.getLogger(name)
    logger.setLevel(level.upper())

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT))
        logger.addHandler(handler)
        logger.propagate = False

    return logger
