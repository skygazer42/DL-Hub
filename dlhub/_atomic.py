"""Internal crash-resistant file primitives for experiment artifacts."""

from __future__ import annotations

from collections.abc import Callable
import os
from pathlib import Path
import secrets
import stat
import time
from typing import Any, BinaryIO

try:
    import msvcrt as _msvcrt
except ImportError:  # pragma: no cover - the native Windows job covers this branch
    _msvcrt: Any | None = None


_WINDOWS = os.name == "nt"
_WINDOWS_REPLACE_TIMEOUT_SECONDS = 2.0
_WINDOWS_REPLACE_RETRY_SECONDS = 0.01


def _sync_directory(path: Path) -> None:
    """Best-effort directory sync after a rename on platforms that support it."""

    if os.name == "nt":
        return

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        directory_fd = os.open(path, flags)
    except OSError:
        return

    try:
        try:
            os.fsync(directory_fd)
        except OSError:
            # Some filesystems do not support syncing directory descriptors.
            pass
    finally:
        try:
            os.close(directory_fd)
        except OSError:
            pass


def _create_temporary(target: Path) -> tuple[int, Path]:
    """Securely create a sibling file using normal create/umask semantics."""

    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0)
    prefix = target.name[:48] or "artifact"
    for _ in range(100):
        temporary = target.parent / f".{prefix}.{secrets.token_hex(8)}.tmp"
        try:
            return os.open(temporary, flags, 0o666), temporary
        except FileExistsError:
            continue
    raise FileExistsError(f"Could not allocate a unique temporary file beside {target}")


def _preserve_mode(file_descriptor: int, temporary: Path, target: Path) -> None:
    """Copy an existing target's basic mode bits onto its replacement."""

    try:
        mode = stat.S_IMODE(target.stat().st_mode)
    except FileNotFoundError:
        return

    if hasattr(os, "fchmod"):
        os.fchmod(file_descriptor, mode)
    else:
        # Windows has no os.fchmod; os.chmod preserves its supported mode bits.
        os.chmod(temporary, mode)


def _replace(temporary: Path, target: Path) -> None:
    """Replace ``target``, tolerating transient Windows sharing violations."""

    deadline = time.monotonic() + _WINDOWS_REPLACE_TIMEOUT_SECONDS
    while True:
        try:
            os.replace(temporary, target)
            return
        except PermissionError:
            # A Windows reader that did not request delete sharing can make an
            # otherwise valid atomic replacement fail with WinError 5.  Keep
            # the fully-synced sibling temporary file and retry for a short,
            # bounded window; permanent permission errors still propagate.
            if not _WINDOWS or time.monotonic() >= deadline:
                raise
            time.sleep(_WINDOWS_REPLACE_RETRY_SECONDS)


def _lock_windows_append(file_descriptor: int) -> None:
    """Take a cross-process one-byte mutex before a Windows append."""

    if _msvcrt is None:  # pragma: no cover - defensive platform invariant
        raise RuntimeError("Windows append locking requires the msvcrt module")
    os.lseek(file_descriptor, 0, os.SEEK_SET)
    _msvcrt.locking(file_descriptor, _msvcrt.LK_LOCK, 1)


def _unlock_windows_append(file_descriptor: int) -> None:
    if _msvcrt is None:  # pragma: no cover - defensive platform invariant
        raise RuntimeError("Windows append locking requires the msvcrt module")
    os.lseek(file_descriptor, 0, os.SEEK_SET)
    _msvcrt.locking(file_descriptor, _msvcrt.LK_UNLCK, 1)


def atomic_write(path: str | Path, writer: Callable[[BinaryIO], None]) -> Path:
    """Write a file completely, then atomically replace its destination.

    The temporary file lives beside the destination so ``os.replace`` stays on
    one filesystem. The old destination remains intact until serialization and
    file syncing have both succeeded.
    """

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary = _create_temporary(target)
    handle: BinaryIO | None = None

    try:
        _preserve_mode(file_descriptor, temporary, target)
        handle = os.fdopen(file_descriptor, "wb")
        with handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())

        _replace(temporary, target)
        _sync_directory(target.parent)
    except BaseException:
        if handle is None:
            os.close(file_descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise

    return target


def append_bytes(path: str | Path, payload: bytes) -> Path:
    """Append one byte record with a single write and sync before returning."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY | getattr(os, "O_BINARY", 0)
    file_descriptor = os.open(target, flags, 0o666)
    windows_lock_held = False
    try:
        # Windows' CRT append mode does not make the seek/write pair atomic
        # across processes.  All DL-Hub writers cooperate on byte zero as a
        # mutex, while POSIX keeps its single O_APPEND write path unchanged.
        if _WINDOWS:
            _lock_windows_append(file_descriptor)
            windows_lock_held = True
        written = os.write(file_descriptor, payload)
        if written != len(payload):
            raise OSError(f"Short append to {target}: wrote {written} of {len(payload)} bytes")
        os.fsync(file_descriptor)
    finally:
        if windows_lock_held:
            _unlock_windows_append(file_descriptor)
        os.close(file_descriptor)

    _sync_directory(target.parent)
    return target


__all__: list[str] = []
