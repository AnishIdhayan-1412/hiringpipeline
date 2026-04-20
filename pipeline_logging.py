# -*- coding: utf-8 -*-
"""
pipeline_logging.py — Centralised Logging Configuration
=========================================================
Bias-Free Hiring Pipeline

Single source of truth for all logging setup across the pipeline.
Import and call ``configure_logging()`` once at application startup;
every subsequent ``logging.getLogger(__name__)`` call in any module
automatically inherits the handlers and formatter defined here.

Usage
-----
    # In main.py or the Flask app factory:
    from pipeline_logging import configure_logging
    configure_logging(verbose=args.verbose, log_dir="logs")

    # In every other module (no change needed):
    import logging
    logger = logging.getLogger(__name__)

Design decisions
----------------
- One rotating file handler (10 MB × 5 backups) so the log directory
  never grows unbounded on a long-running server.
- One stream handler for the console so operators see live output.
- Coloured console output (ANSI) when stdout is a TTY.
- JSON-structured file output so log aggregators (Datadog, Loki, etc.)
  can ingest the logs without a custom parser.
- The root logger is configured exactly once; re-entrant calls are
  silently ignored via a module-level ``_configured`` guard.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import sys
from datetime import datetime, timezone
from typing import Optional


# ── Guard: prevent double-configuration ───────────────────────────────────
_configured: bool = False

# ── Default values ────────────────────────────────────────────────────────
DEFAULT_LOG_DIR:    str = "logs"
DEFAULT_LOG_FILE:   str = "pipeline.log"
MAX_BYTES:          int = 10 * 1024 * 1024   # 10 MB per file
BACKUP_COUNT:       int = 5
CONSOLE_FMT:        str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
DATE_FMT:           str = "%H:%M:%S"

# ── ANSI colour codes for TTY consoles ────────────────────────────────────
_COLOURS: dict[str, str] = {
    "DEBUG":    "\033[36m",   # cyan
    "INFO":     "\033[32m",   # green
    "WARNING":  "\033[33m",   # amber
    "ERROR":    "\033[31m",   # red
    "CRITICAL": "\033[35m",   # magenta
    "RESET":    "\033[0m",
}


class _ColourFormatter(logging.Formatter):
    """Console formatter that adds ANSI colour to the level name.

    Falls back to plain text when stdout is not a TTY (e.g. Docker logs
    piped to a file) so colour escape codes never appear in plain-text files.
    """

    def __init__(self, fmt: str, datefmt: str, use_colour: bool = True) -> None:
        super().__init__(fmt=fmt, datefmt=datefmt)
        self._use_colour = use_colour

    def format(self, record: logging.LogRecord) -> str:
        if self._use_colour:
            colour = _COLOURS.get(record.levelname, "")
            reset  = _COLOURS["RESET"]
            record.levelname = f"{colour}{record.levelname:<8}{reset}"
        return super().format(record)


class _JsonFormatter(logging.Formatter):
    """File formatter that emits one JSON object per log line.

    Each line is a complete, parseable JSON object with a standard set
    of fields, making it trivial to ingest into any log aggregator.

    Example output::

        {"ts":"2024-01-15T14:32:01Z","level":"INFO","logger":"module1",
         "msg":"SBERT model loaded successfully.","pid":1234}
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
            "ts":     datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "level":  record.levelname,
            "logger": record.name,
            "msg":    record.getMessage(),
            "pid":    record.process,
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(
    verbose:  bool            = False,
    log_dir:  Optional[str]   = DEFAULT_LOG_DIR,
    log_file: str             = DEFAULT_LOG_FILE,
) -> None:
    """Configure the root logger for the entire pipeline.

    This function is idempotent — calling it more than once has no effect,
    so it is safe to call from both ``main.py`` and ``module5.py``.

    Args:
        verbose:  If ``True``, sets the root level to ``DEBUG``; otherwise
                  ``INFO``.  Individual modules can still override their own
                  logger level if needed.
        log_dir:  Directory where the rotating log file will be written.
                  Pass ``None`` to disable file logging (e.g. in unit tests).
        log_file: Base filename for the rotating log (default: pipeline.log).
    """
    global _configured
    if _configured:
        return
    _configured = True

    root    = logging.getLogger()
    level   = logging.DEBUG if verbose else logging.INFO
    root.setLevel(level)

    # Remove any handlers that basicConfig may have added before this call
    root.handlers.clear()

    # ── Console handler ───────────────────────────────────────────────────
    use_colour = sys.stdout.isatty() and os.name != "nt"
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(
        _ColourFormatter(fmt=CONSOLE_FMT, datefmt=DATE_FMT, use_colour=use_colour)
    )
    root.addHandler(console_handler)

    # ── Rotating file handler (JSON) ──────────────────────────────────────
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, log_file)
        file_handler = logging.handlers.RotatingFileHandler(
            filename    = log_path,
            maxBytes    = MAX_BYTES,
            backupCount = BACKUP_COUNT,
            encoding    = "utf-8",
        )
        file_handler.setLevel(logging.DEBUG)   # always capture DEBUG in file
        file_handler.setFormatter(_JsonFormatter())
        root.addHandler(file_handler)

    # Silence noisy third-party loggers at WARNING unless verbose
    if not verbose:
        for noisy in ("urllib3", "filelock", "sentence_transformers",
                      "transformers", "torch", "PIL", "werkzeug"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

    logging.getLogger(__name__).debug(
        "Logging configured — level=%s, file=%s",
        "DEBUG" if verbose else "INFO",
        os.path.join(log_dir or "", log_file) if log_dir else "disabled",
    )


def reset_logging() -> None:
    """Reset the logging configuration (intended for unit tests only).

    Clears all root handlers and resets the ``_configured`` guard so that
    ``configure_logging()`` can be called again in a fresh test environment.
    """
    global _configured
    _configured = False
    root = logging.getLogger()
    root.handlers.clear()
