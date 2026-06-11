"""
Centralized logging configuration.

Sets up console + file handlers with rotation, standard format, and
level overrides for noisy third-party libraries.
"""

import logging
import logging.handlers
from pathlib import Path


def setup_logging(level: str = "INFO", log_dir: str = "logs") -> None:
    """Configure root logger with console and rotating-file handlers.

    Called once at application startup (main.py).
    Existing handlers are cleared first, so calling this multiple times is
    safe (idempotent).
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Clear existing handlers to make repeated calls safe
    root.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(name)s:%(lineno)d  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler — use the effective level (respect root level)
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    console.setLevel(root.level)
    root.addHandler(console)

    # Rotating file handler — always DEBUG to disk
    file_handler = logging.handlers.RotatingFileHandler(
        log_path / "app.log",
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(fmt)
    file_handler.setLevel(logging.DEBUG)
    root.addHandler(file_handler)

    # Silence noisy third-party loggers
    for noisy in ("httpx", "httpcore", "openai", "chromadb", "sqlalchemy.engine"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # Let application loggers propagate to root handlers
    logging.getLogger("backend").setLevel(logging.DEBUG)
