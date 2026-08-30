import logging
import os
from pathlib import Path
import threading


_HANDLER_KIND = "_dlhub_handler_kind"
_CONFIG_LOCK = threading.RLock()


def _file_identity(path: str | Path) -> str:
    return os.path.normcase(os.path.abspath(os.fspath(path)))


def _is_managed_file_handler(handler: logging.Handler) -> bool:
    return isinstance(handler, logging.FileHandler) and getattr(
        handler, _HANDLER_KIND, None
    ) == "file"


def _points_to(handler: logging.FileHandler, path: Path) -> bool:
    return _file_identity(handler.baseFilename) == _file_identity(path)


def get_logger(name: str = "dlhub", log_file: str | Path | None = None) -> logging.Logger:
    """Return an INFO logger, optionally routing it to exactly one managed file.

    Repeated calls are idempotent. When the same logger name is reused for a new
    run, its DL-Hub-managed file handler is moved to the new file instead of
    continuing to write into the previous run's log. Handlers installed by the
    application are preserved.
    """

    log_path = Path(log_file).expanduser() if log_file is not None else None

    with _CONFIG_LOCK:
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")

        # Preserve the historical behavior of adding a console only when the
        # application has not configured this named logger itself.
        if not logger.handlers:
            console = logging.StreamHandler()
            setattr(console, _HANDLER_KIND, "console")
            console.setFormatter(formatter)
            logger.addHandler(console)

        if log_path is None:
            return logger

        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handlers = [
            handler for handler in logger.handlers if isinstance(handler, logging.FileHandler)
        ]
        matching_handler = next(
            (handler for handler in file_handlers if _points_to(handler, log_path)),
            None,
        )

        # Open the replacement before detaching the old handler. A permission or
        # filesystem failure therefore leaves the current logger usable.
        replacement: logging.FileHandler | None = None
        if matching_handler is None:
            replacement = logging.FileHandler(log_path, encoding="utf-8")
            setattr(replacement, _HANDLER_KIND, "file")
            replacement.setFormatter(formatter)

        for handler in file_handlers:
            if _is_managed_file_handler(handler) and handler is not matching_handler:
                logger.removeHandler(handler)
                handler.close()

        if replacement is not None:
            logger.addHandler(replacement)

        return logger
