"""Logging configuration using rich for readable console output."""

import logging

from rich.console import Console
from rich.logging import RichHandler

_NOISY_LOGGERS = ("httpx", "chromadb", "sentence_transformers", "urllib3")


def setup_logging(level: str = "INFO") -> None:
    """Configure root logging with a Rich handler and silence noisy third-party loggers."""
    # legacy_windows=False avoids rich's raw Win32 console API, which encodes
    # using the console's codepage (e.g. cp1252) and can crash a --verbose run
    # that logs non-ASCII text - see docs/decisions.md.
    console = Console(stderr=True, legacy_windows=False)
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)],
    )
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
