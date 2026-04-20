# -*- coding: utf-8 -*-
"""Tests for pipeline_logging.py — verifies idempotency and handler setup."""
import logging
import pipeline_logging


def test_configure_logging_idempotent(tmp_path):
    """Calling configure_logging twice must not add duplicate handlers."""
    pipeline_logging.reset_logging()
    pipeline_logging.configure_logging(verbose=False, log_dir=str(tmp_path))
    handlers_after_first = len(logging.getLogger().handlers)

    pipeline_logging.configure_logging(verbose=False, log_dir=str(tmp_path))
    handlers_after_second = len(logging.getLogger().handlers)

    assert handlers_after_first == handlers_after_second
    pipeline_logging.reset_logging()


def test_configure_logging_creates_log_file(tmp_path):
    """A log file should be created in log_dir after configure_logging."""
    pipeline_logging.reset_logging()
    pipeline_logging.configure_logging(verbose=False, log_dir=str(tmp_path))
    log_file = tmp_path / "pipeline.log"
    logging.getLogger("test_logger").info("hello from test")
    # Flush all handlers
    for h in logging.getLogger().handlers:
        h.flush()
    assert log_file.exists()
    pipeline_logging.reset_logging()


def test_reset_logging_clears_handlers():
    pipeline_logging.reset_logging()
    pipeline_logging.configure_logging(verbose=True, log_dir=None)
    assert len(logging.getLogger().handlers) > 0
    pipeline_logging.reset_logging()
    assert len(logging.getLogger().handlers) == 0
