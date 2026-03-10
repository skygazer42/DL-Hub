from __future__ import annotations

import math

import torch


def _validate_boxes_scores(boxes: torch.Tensor, scores: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if not torch.is_tensor(boxes):
        raise TypeError(f"boxes must be a torch.Tensor, got {type(boxes)!r}")
    if not torch.is_tensor(scores):
        raise TypeError(f"scores must be a torch.Tensor, got {type(scores)!r}")

    if boxes.ndim != 2 or boxes.shape[-1] != 4:
        raise ValueError(f"boxes must have shape (N, 4), got {tuple(boxes.shape)}")
    if scores.ndim != 1:
        raise ValueError(f"scores must have shape (N,), got {tuple(scores.shape)}")
    if int(scores.shape[0]) != int(boxes.shape[0]):
        raise ValueError(
            f"boxes and scores must have the same length, got N={int(boxes.shape[0])} and "
            f"scores={int(scores.shape[0])}"
        )

    return boxes.to(torch.float32), scores.to(torch.float32)


def _box_iou_xyxy(box: torch.Tensor, boxes: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    """IoU between one box (4,) and many boxes (N,4), all in xyxy."""

    box = box.to(torch.float32)
    boxes = boxes.to(torch.float32)

    ix1 = torch.maximum(box[0], boxes[:, 0])
    iy1 = torch.maximum(box[1], boxes[:, 1])
    ix2 = torch.minimum(box[2], boxes[:, 2])
    iy2 = torch.minimum(box[3], boxes[:, 3])

    iw = (ix2 - ix1).clamp(min=0.0)
    ih = (iy2 - iy1).clamp(min=0.0)
    inter = iw * ih

    area_box = (box[2] - box[0]).clamp(min=0.0) * (box[3] - box[1]).clamp(min=0.0)
    area_boxes = (boxes[:, 2] - boxes[:, 0]).clamp(min=0.0) * (boxes[:, 3] - boxes[:, 1]).clamp(
        min=0.0
    )
    union = (area_box + area_boxes - inter).clamp(min=float(eps))
    return inter / union


def _box_diou_xyxy(box: torch.Tensor, boxes: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    """DIoU between one box (4,) and many boxes (N,4), all in xyxy."""

    iou = _box_iou_xyxy(box, boxes, eps=float(eps))

    cx = (box[0] + box[2]) * 0.5
    cy = (box[1] + box[3]) * 0.5
    cxs = (boxes[:, 0] + boxes[:, 2]) * 0.5
    cys = (boxes[:, 1] + boxes[:, 3]) * 0.5
    rho2 = (cxs - cx).square() + (cys - cy).square()

    ex1 = torch.minimum(box[0], boxes[:, 0])
    ey1 = torch.minimum(box[1], boxes[:, 1])
    ex2 = torch.maximum(box[2], boxes[:, 2])
    ey2 = torch.maximum(box[3], boxes[:, 3])
    c2 = (ex2 - ex1).square() + (ey2 - ey1).square()
    c2 = c2.clamp(min=float(eps))

    return iou - rho2 / c2


def soft_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    *,
    iou_threshold: float = 0.5,
    sigma: float = 0.5,
    score_threshold: float = 1e-3,
    method: str = "gaussian",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Soft-NMS for xyxy boxes.

    This is a toy-first implementation intended for small-N post-processing and educational use.

    Args:
        boxes: (N, 4) in xyxy.
        scores: (N,) confidence scores.
        iou_threshold: IoU threshold used for the linear/hard methods.
        sigma: Gaussian sigma used for the gaussian method.
        score_threshold: boxes with final score <= threshold are dropped from `keep`.
        method: "gaussian" | "linear" | "hard".

    Returns:
        keep_indices: int64 tensor of kept box indices, sorted by final score desc.
        scores_out: (N,) tensor of updated scores (in original input order).
    """

    boxes, scores = _validate_boxes_scores(boxes, scores)
    n = int(boxes.shape[0])
    if n == 0:
        empty = torch.empty((0,), device=boxes.device, dtype=torch.int64)
        return empty, scores

    method = str(method).lower().strip()
    if method not in {"gaussian", "linear", "hard"}:
        raise ValueError("method must be one of: gaussian | linear | hard")

    iou_threshold = float(iou_threshold)
    if not math.isfinite(iou_threshold):
        raise ValueError("iou_threshold must be finite")

    sigma = float(sigma)
    if method == "gaussian" and (not math.isfinite(sigma) or sigma <= 0.0):
        raise ValueError("sigma must be finite and > 0 for gaussian soft-nms")

    score_threshold = float(score_threshold)

    boxes_work = boxes.clone()
    scores_work = scores.clone()
    indices = torch.arange(n, device=boxes.device, dtype=torch.int64)

    for i in range(n):
        # Select best remaining box.
        max_pos = i + int(scores_work[i:].argmax().item())
        if max_pos != i:
            boxes_work[[i, max_pos]] = boxes_work[[max_pos, i]]
            scores_work[[i, max_pos]] = scores_work[[max_pos, i]]
            indices[[i, max_pos]] = indices[[max_pos, i]]

        if i == n - 1:
            break

        ious = _box_iou_xyxy(boxes_work[i], boxes_work[i + 1 :])
        if method == "linear":
            decay = torch.ones_like(ious)
            mask = ious > iou_threshold
            decay[mask] = 1.0 - ious[mask]
            scores_work[i + 1 :] = scores_work[i + 1 :] * decay
        elif method == "gaussian":
            scores_work[i + 1 :] = scores_work[i + 1 :] * torch.exp(-(ious * ious) / sigma)
        else:  # hard
            scores_work[i + 1 :][ious > iou_threshold] = 0.0

    scores_out = scores.clone()
    scores_out[indices] = scores_work

    keep = torch.nonzero(scores_out > score_threshold, as_tuple=False).squeeze(1).to(torch.int64)
    if keep.numel() == 0:
        return keep, scores_out

    keep = keep[scores_out[keep].argsort(descending=True)]
    return keep, scores_out


def diou_nms(boxes: torch.Tensor, scores: torch.Tensor, *, threshold: float = 0.5) -> torch.Tensor:
    """DIoU-NMS for xyxy boxes.

    Same loop structure as standard NMS, but uses DIoU as the suppression metric.

    Args:
        boxes: (N,4) in xyxy.
        scores: (N,)
        threshold: suppress boxes with DIoU > threshold.

    Returns:
        keep_indices: int64 tensor of kept indices (sorted by descending score).
    """

    boxes, scores = _validate_boxes_scores(boxes, scores)
    n = int(boxes.shape[0])
    if n == 0:
        return torch.empty((0,), device=boxes.device, dtype=torch.int64)

    thr = float(threshold)
    if not math.isfinite(thr):
        raise ValueError("threshold must be finite")

    order = scores.argsort(descending=True)
    keep: list[torch.Tensor] = []

    while int(order.numel()) > 0:
        i = order[0]
        keep.append(i)
        if int(order.numel()) == 1:
            break

        rest = order[1:]
        diou = _box_diou_xyxy(boxes[i], boxes[rest])
        order = rest[diou <= thr]

    return torch.stack(keep).to(torch.int64)


__all__ = ["diou_nms", "soft_nms"]

