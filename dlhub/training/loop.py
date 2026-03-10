from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
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

    model.train()
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
            for hook in hooks:
                hook.on_batch_end(log)

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

    model.eval()
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
                for hook in hooks:
                    hook.on_batch_end(log)

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

    model.train()
    total_loss = 0.0
    correct = 0
    total_tokens = 0

    for batch_idx, (inputs, targets) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        inputs = to_device(inputs, device=device)
        targets = to_device(targets, device=device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)  # (B, T, C)
        if logits.ndim != 3:
            raise ValueError(f"Expected logits shape (B, T, C), got {tuple(logits.shape)}")

        b, t, c = logits.shape
        loss = criterion(logits.reshape(b * t, c), targets.reshape(b * t))
        loss.backward()
        optimizer.step()

        mask = targets != int(ignore_index)
        batch_tokens = int(mask.sum().item())
        if batch_tokens > 0:
            with torch.no_grad():
                pred = logits.argmax(dim=-1)
                batch_correct = int(((pred == targets) & mask).sum().item())
            correct += batch_correct
            total_tokens += batch_tokens
            total_loss += float(loss.item()) * batch_tokens
        else:
            batch_correct = 0

        if hooks:
            batch_loss = float(loss.item())
            batch_acc = batch_correct / batch_tokens if batch_tokens else 0.0
            log = BatchLog(stage="train", batch_idx=batch_idx, loss=batch_loss, accuracy=batch_acc)
            for hook in hooks:
                hook.on_batch_end(log)

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

    model.eval()
    total_loss = 0.0
    correct = 0
    total_tokens = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            inputs = to_device(inputs, device=device)
            targets = to_device(targets, device=device)
            logits = model(inputs)
            if logits.ndim != 3:
                raise ValueError(f"Expected logits shape (B, T, C), got {tuple(logits.shape)}")

            b, t, c = logits.shape
            loss = criterion(logits.reshape(b * t, c), targets.reshape(b * t))

            mask = targets != int(ignore_index)
            batch_tokens = int(mask.sum().item())
            if batch_tokens > 0:
                pred = logits.argmax(dim=-1)
                batch_correct = int(((pred == targets) & mask).sum().item())
                correct += batch_correct
                total_tokens += batch_tokens
                total_loss += float(loss.item()) * batch_tokens
            else:
                batch_correct = 0

            if hooks:
                batch_loss = float(loss.item())
                batch_acc = batch_correct / batch_tokens if batch_tokens else 0.0
                log = BatchLog(
                    stage="eval", batch_idx=batch_idx, loss=batch_loss, accuracy=batch_acc
                )
                for hook in hooks:
                    hook.on_batch_end(log)

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

    model.train()
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
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        batch_loss = float(loss.item())
        batch_size = int(targets.size(0))
        total += batch_size
        total_loss += batch_loss * batch_size

        with torch.no_grad():
            probs = torch.sigmoid(logits)
            preds = (probs > float(threshold)).to(torch.float32)
            intersection = (preds * targets).sum(dim=(1, 2, 3))
            union = (preds + targets - preds * targets).sum(dim=(1, 2, 3))
            iou = (intersection / union.clamp_min(1e-6)).mean()
            total_iou += float(iou.item()) * batch_size

        if hooks:
            log = BatchLog(
                stage="train", batch_idx=batch_idx, loss=batch_loss, accuracy=float(iou.item())
            )
            for hook in hooks:
                hook.on_batch_end(log)

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

    model.eval()
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
            loss = criterion(logits, targets)

            batch_loss = float(loss.item())
            batch_size = int(targets.size(0))
            total += batch_size
            total_loss += batch_loss * batch_size

            probs = torch.sigmoid(logits)
            preds = (probs > float(threshold)).to(torch.float32)
            intersection = (preds * targets).sum(dim=(1, 2, 3))
            union = (preds + targets - preds * targets).sum(dim=(1, 2, 3))
            iou = (intersection / union.clamp_min(1e-6)).mean()
            total_iou += float(iou.item()) * batch_size

            if hooks:
                log = BatchLog(
                    stage="eval", batch_idx=batch_idx, loss=batch_loss, accuracy=float(iou.item())
                )
                for hook in hooks:
                    hook.on_batch_end(log)

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
    import torch

    model.train()
    total_loss = 0.0
    total = 0

    def infer_batch_size(x) -> int | None:
        if torch.is_tensor(x):
            return int(x.shape[0]) if x.ndim >= 1 else 1
        if isinstance(x, dict):
            for v in x.values():
                bs = infer_batch_size(v)
                if bs is not None:
                    return bs
            return None
        if isinstance(x, list | tuple):
            for v in x:
                bs = infer_batch_size(v)
                if bs is not None:
                    return bs
            return None
        return None

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
        batch_size = infer_batch_size(targets)
        if batch_size is None:
            batch_size = infer_batch_size(inputs)
        if batch_size is None:
            batch_size = 0
        total += batch_size
        total_loss += batch_loss * batch_size
        if hooks:
            log = BatchLog(stage="train", batch_idx=batch_idx, loss=batch_loss, accuracy=None)
            for hook in hooks:
                hook.on_batch_end(log)

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

    model.eval()
    total_loss = 0.0
    total = 0

    def infer_batch_size(x) -> int | None:
        if torch.is_tensor(x):
            return int(x.shape[0]) if x.ndim >= 1 else 1
        if isinstance(x, dict):
            for v in x.values():
                bs = infer_batch_size(v)
                if bs is not None:
                    return bs
            return None
        if isinstance(x, list | tuple):
            for v in x:
                bs = infer_batch_size(v)
                if bs is not None:
                    return bs
            return None
        return None

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            inputs = to_device(inputs, device=device)
            targets = to_device(targets, device=device)
            preds = model(inputs)
            loss = criterion(preds, targets)

            batch_loss = float(loss.item())
            batch_size = infer_batch_size(targets)
            if batch_size is None:
                batch_size = infer_batch_size(inputs)
            if batch_size is None:
                batch_size = 0
            total += batch_size
            total_loss += batch_loss * batch_size
            if hooks:
                log = BatchLog(stage="eval", batch_idx=batch_idx, loss=batch_loss, accuracy=None)
                for hook in hooks:
                    hook.on_batch_end(log)

    avg_loss = total_loss / total if total else 0.0
    return RegressionStats(loss=avg_loss)
