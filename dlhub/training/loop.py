from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import operator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

from .batch import to_device
from .hooks import BatchLog, Hook


@dataclass(frozen=True)
class TrainStats:
    loss: float
    accuracy: float


@dataclass(frozen=True)
class RegressionStats:
    loss: float


@dataclass(frozen=True)
class TokenStats:
    loss: float
    accuracy: float


@dataclass(frozen=True)
class SegmentationStats:
    loss: float
    iou: float


def _validate_max_batches(max_batches: int | None) -> int | None:
    if max_batches is None:
        return None
    if isinstance(max_batches, bool):
        raise TypeError("max_batches must be a non-negative integer or None, not bool")
    try:
        value = operator.index(max_batches)
    except TypeError as exc:
        raise TypeError("max_batches must be a non-negative integer or None") from exc
    if value < 0:
        raise ValueError(f"max_batches must be >= 0, got {value}")
    return value


def _notify_hooks(hooks: Sequence[Hook], log: BatchLog) -> None:
    for hook in hooks:
        try:
            hook.on_batch_end(log)
        except Exception as exc:
            raise RuntimeError(
                f"Hook {type(hook).__name__} failed at "
                f"stage={log.stage!r}, batch_idx={log.batch_idx}"
            ) from exc


def _validate_segmentation_shapes(logits: torch.Tensor, targets: torch.Tensor) -> None:
    if logits.shape != targets.shape:
        raise ValueError(
            f"Binary segmentation logits and targets must have the same shape, "
            f"got {tuple(logits.shape)} and {tuple(targets.shape)}"
        )
    if logits.ndim < 2:
        raise ValueError(
            f"Binary segmentation tensors must include batch and feature dimensions, "
            f"got {tuple(logits.shape)}"
        )


def _binary_iou(logits: torch.Tensor, targets: torch.Tensor, *, threshold: float) -> torch.Tensor:
    probs = logits.sigmoid()
    preds = (probs > float(threshold)).to(targets.dtype)
    reduce_dims = tuple(range(1, preds.ndim))
    intersection = (preds * targets).sum(dim=reduce_dims)
    union = (preds + targets - preds * targets).sum(dim=reduce_dims)
    per_sample = intersection / union.clamp_min(1e-12)
    # Two empty sets are an exact match. This also avoids penalizing datasets
    # that contain valid negative-only masks.
    return per_sample.masked_fill(union == 0, 1.0).mean()


def _infer_batch_size(value: object) -> int | None:
    import torch

    if torch.is_tensor(value):
        return int(value.shape[0]) if value.ndim >= 1 else 1
    if isinstance(value, Mapping):
        values = value.values()
    elif isinstance(value, list | tuple):
        values = value
    else:
        return None

    for nested in values:
        batch_size = _infer_batch_size(nested)
        if batch_size is not None:
            return batch_size
    return None


def fit_classifier(
    *,
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    max_batches: int | None = None,
    hooks: Sequence[Hook] | None = None,
) -> TrainStats:
    import torch

    max_batches = _validate_max_batches(max_batches)
    model.train()
    if max_batches == 0:
        return TrainStats(loss=0.0, accuracy=0.0)
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        inputs = to_device(inputs, device=device)
        targets = to_device(targets, device=device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        batch_loss = float(loss.item())
        batch_size = targets.size(0)
        total += batch_size
        total_loss += batch_loss * batch_size

        batch_correct = int((torch.argmax(logits, dim=1) == targets).sum().item())
        correct += batch_correct
        if hooks:
            batch_acc = batch_correct / batch_size if batch_size else 0.0
            log = BatchLog(stage="train", batch_idx=batch_idx, loss=batch_loss, accuracy=batch_acc)
            _notify_hooks(hooks, log)

    avg_loss = total_loss / total if total else 0.0
    acc = correct / total if total else 0.0
    return TrainStats(loss=avg_loss, accuracy=acc)


def evaluate_classifier(
    *,
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    max_batches: int | None = None,
    hooks: Sequence[Hook] | None = None,
) -> TrainStats:
    import torch

    max_batches = _validate_max_batches(max_batches)
    model.eval()
    if max_batches == 0:
        return TrainStats(loss=0.0, accuracy=0.0)
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            inputs = to_device(inputs, device=device)
            targets = to_device(targets, device=device)
            logits = model(inputs)
            loss = criterion(logits, targets)

            batch_loss = float(loss.item())
            batch_size = targets.size(0)
            total += batch_size
            total_loss += batch_loss * batch_size

            batch_correct = int((torch.argmax(logits, dim=1) == targets).sum().item())
            correct += batch_correct
            if hooks:
                batch_acc = batch_correct / batch_size if batch_size else 0.0
                log = BatchLog(
                    stage="eval", batch_idx=batch_idx, loss=batch_loss, accuracy=batch_acc
                )
                _notify_hooks(hooks, log)

    avg_loss = total_loss / total if total else 0.0
    acc = correct / total if total else 0.0
    return TrainStats(loss=avg_loss, accuracy=acc)


def fit_token_classifier(
    *,
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    max_batches: int | None = None,
    ignore_index: int = -100,
    hooks: Sequence[Hook] | None = None,
) -> TokenStats:
    import torch

    max_batches = _validate_max_batches(max_batches)
    model.train()
    if max_batches == 0:
        return TokenStats(loss=0.0, accuracy=0.0)
    total_loss = 0.0
    correct = 0
    total_tokens = 0

    for batch_idx, (inputs, targets) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        inputs = to_device(inputs, device=device)
        targets = to_device(targets, device=device)

        optimizer.zero_grad(set_to_none=True)
        mask = targets != int(ignore_index)
        batch_tokens = int(mask.sum().item())
        if batch_tokens == 0:
            if hooks:
                _notify_hooks(
                    hooks,
                    BatchLog(stage="train", batch_idx=batch_idx, loss=0.0, accuracy=0.0),
                )
            continue

        logits = model(inputs)  # (B, T, C)
        if logits.ndim != 3:
            raise ValueError(f"Expected logits shape (B, T, C), got {tuple(logits.shape)}")

        b, t, c = logits.shape
        loss = criterion(logits.reshape(b * t, c), targets.reshape(b * t))
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            pred = logits.argmax(dim=-1)
            batch_correct = int(((pred == targets) & mask).sum().item())
        batch_loss = float(loss.item())
        correct += batch_correct
        total_tokens += batch_tokens
        total_loss += batch_loss * batch_tokens

        if hooks:
            batch_acc = batch_correct / batch_tokens
            log = BatchLog(stage="train", batch_idx=batch_idx, loss=batch_loss, accuracy=batch_acc)
            _notify_hooks(hooks, log)

    avg_loss = total_loss / total_tokens if total_tokens else 0.0
    acc = correct / total_tokens if total_tokens else 0.0
    return TokenStats(loss=avg_loss, accuracy=acc)


def evaluate_token_classifier(
    *,
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    max_batches: int | None = None,
    ignore_index: int = -100,
    hooks: Sequence[Hook] | None = None,
) -> TokenStats:
    import torch

    max_batches = _validate_max_batches(max_batches)
    model.eval()
    if max_batches == 0:
        return TokenStats(loss=0.0, accuracy=0.0)
    total_loss = 0.0
    correct = 0
    total_tokens = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            inputs = to_device(inputs, device=device)
            targets = to_device(targets, device=device)
            mask = targets != int(ignore_index)
            batch_tokens = int(mask.sum().item())
            if batch_tokens == 0:
                if hooks:
                    _notify_hooks(
                        hooks,
                        BatchLog(stage="eval", batch_idx=batch_idx, loss=0.0, accuracy=0.0),
                    )
                continue

            logits = model(inputs)
            if logits.ndim != 3:
                raise ValueError(f"Expected logits shape (B, T, C), got {tuple(logits.shape)}")

            b, t, c = logits.shape
            loss = criterion(logits.reshape(b * t, c), targets.reshape(b * t))

            pred = logits.argmax(dim=-1)
            batch_correct = int(((pred == targets) & mask).sum().item())
            batch_loss = float(loss.item())
            correct += batch_correct
            total_tokens += batch_tokens
            total_loss += batch_loss * batch_tokens

            if hooks:
                batch_acc = batch_correct / batch_tokens
                log = BatchLog(
                    stage="eval", batch_idx=batch_idx, loss=batch_loss, accuracy=batch_acc
                )
                _notify_hooks(hooks, log)

    avg_loss = total_loss / total_tokens if total_tokens else 0.0
    acc = correct / total_tokens if total_tokens else 0.0
    return TokenStats(loss=avg_loss, accuracy=acc)


def fit_binary_segmentation(
    *,
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    max_batches: int | None = None,
    threshold: float = 0.5,
    hooks: Sequence[Hook] | None = None,
) -> SegmentationStats:
    import torch

    max_batches = _validate_max_batches(max_batches)
    model.train()
    if max_batches == 0:
        return SegmentationStats(loss=0.0, iou=0.0)
    total_loss = 0.0
    total_iou = 0.0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        inputs = to_device(inputs, device=device)
        targets = to_device(targets, device=device).to(torch.float32)

        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        _validate_segmentation_shapes(logits, targets)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        batch_loss = float(loss.item())
        batch_size = int(targets.size(0))
        total += batch_size
        total_loss += batch_loss * batch_size

        with torch.no_grad():
            iou = _binary_iou(logits, targets, threshold=threshold)
            total_iou += float(iou.item()) * batch_size

        if hooks:
            log = BatchLog(
                stage="train", batch_idx=batch_idx, loss=batch_loss, accuracy=float(iou.item())
            )
            _notify_hooks(hooks, log)

    avg_loss = total_loss / total if total else 0.0
    avg_iou = total_iou / total if total else 0.0
    return SegmentationStats(loss=avg_loss, iou=avg_iou)


def evaluate_binary_segmentation(
    *,
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    max_batches: int | None = None,
    threshold: float = 0.5,
    hooks: Sequence[Hook] | None = None,
) -> SegmentationStats:
    import torch

    max_batches = _validate_max_batches(max_batches)
    model.eval()
    if max_batches == 0:
        return SegmentationStats(loss=0.0, iou=0.0)
    total_loss = 0.0
    total_iou = 0.0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            inputs = to_device(inputs, device=device)
            targets = to_device(targets, device=device).to(torch.float32)

            logits = model(inputs)
            _validate_segmentation_shapes(logits, targets)
            loss = criterion(logits, targets)

            batch_loss = float(loss.item())
            batch_size = int(targets.size(0))
            total += batch_size
            total_loss += batch_loss * batch_size

            iou = _binary_iou(logits, targets, threshold=threshold)
            total_iou += float(iou.item()) * batch_size

            if hooks:
                log = BatchLog(
                    stage="eval", batch_idx=batch_idx, loss=batch_loss, accuracy=float(iou.item())
                )
                _notify_hooks(hooks, log)

    avg_loss = total_loss / total if total else 0.0
    avg_iou = total_iou / total if total else 0.0
    return SegmentationStats(loss=avg_loss, iou=avg_iou)


def fit_regression(
    *,
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    max_batches: int | None = None,
    hooks: Sequence[Hook] | None = None,
) -> RegressionStats:
    max_batches = _validate_max_batches(max_batches)
    model.train()
    if max_batches == 0:
        return RegressionStats(loss=0.0)
    total_loss = 0.0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        inputs = to_device(inputs, device=device)
        targets = to_device(targets, device=device)

        optimizer.zero_grad(set_to_none=True)
        preds = model(inputs)
        loss = criterion(preds, targets)
        loss.backward()
        optimizer.step()

        batch_loss = float(loss.item())
        batch_size = _infer_batch_size(targets)
        if batch_size is None:
            batch_size = _infer_batch_size(inputs)
        if batch_size is None:
            batch_size = 0
        total += batch_size
        total_loss += batch_loss * batch_size
        if hooks:
            log = BatchLog(stage="train", batch_idx=batch_idx, loss=batch_loss, accuracy=None)
            _notify_hooks(hooks, log)

    avg_loss = total_loss / total if total else 0.0
    return RegressionStats(loss=avg_loss)


def evaluate_regression(
    *,
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    max_batches: int | None = None,
    hooks: Sequence[Hook] | None = None,
) -> RegressionStats:
    import torch

    max_batches = _validate_max_batches(max_batches)
    model.eval()
    if max_batches == 0:
        return RegressionStats(loss=0.0)
    total_loss = 0.0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            inputs = to_device(inputs, device=device)
            targets = to_device(targets, device=device)
            preds = model(inputs)
            loss = criterion(preds, targets)

            batch_loss = float(loss.item())
            batch_size = _infer_batch_size(targets)
            if batch_size is None:
                batch_size = _infer_batch_size(inputs)
            if batch_size is None:
                batch_size = 0
            total += batch_size
            total_loss += batch_loss * batch_size
            if hooks:
                log = BatchLog(stage="eval", batch_idx=batch_idx, loss=batch_loss, accuracy=None)
                _notify_hooks(hooks, log)

    avg_loss = total_loss / total if total else 0.0
    return RegressionStats(loss=avg_loss)
