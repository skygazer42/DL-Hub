import io
import logging
from pathlib import Path
from uuid import uuid4

from dlhub.logging import get_logger


def _fresh_logger_name() -> str:
    return f"dlhub.tests.{uuid4().hex}"


def _close_handlers(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()


def test_get_logger_switches_managed_file_between_runs(tmp_path: Path) -> None:
    name = _fresh_logger_name()
    first_path = tmp_path / "run-one" / "train.log"
    second_path = tmp_path / "run-two" / "train.log"
    logger = get_logger(name, log_file=first_path)

    try:
        logger.info("first-run-only")
        same_logger = get_logger(name, log_file=second_path)
        same_logger.info("second-run-only")

        assert same_logger is logger
        assert "first-run-only" in first_path.read_text(encoding="utf-8")
        assert "second-run-only" not in first_path.read_text(encoding="utf-8")
        assert "second-run-only" in second_path.read_text(encoding="utf-8")

        file_handlers = [
            handler
            for handler in logger.handlers
            if isinstance(handler, logging.FileHandler)
        ]
        assert len(file_handlers) == 1
        assert Path(file_handlers[0].baseFilename) == second_path
    finally:
        _close_handlers(logger)


def test_get_logger_is_idempotent_for_same_file(tmp_path: Path) -> None:
    name = _fresh_logger_name()
    log_path = tmp_path / "train.log"
    logger = get_logger(name, log_file=log_path)

    try:
        get_logger(name, log_file=log_path)
        get_logger(name, log_file=log_path)
        logger.info("written-once")

        file_handlers = [
            handler
            for handler in logger.handlers
            if isinstance(handler, logging.FileHandler)
        ]
        assert len(file_handlers) == 1
        assert log_path.read_text(encoding="utf-8").count("written-once") == 1
    finally:
        _close_handlers(logger)


def test_get_logger_preserves_application_handlers_and_adds_requested_file(
    tmp_path: Path,
) -> None:
    name = _fresh_logger_name()
    logger = logging.getLogger(name)
    stream = io.StringIO()
    application_handler = logging.StreamHandler(stream)
    logger.addHandler(application_handler)
    log_path = tmp_path / "train.log"

    try:
        configured = get_logger(name, log_file=log_path)
        configured.info("shared-message")

        assert application_handler in configured.handlers
        assert "shared-message" in stream.getvalue()
        assert "shared-message" in log_path.read_text(encoding="utf-8")
    finally:
        _close_handlers(logger)
