from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass

import torch

from dlhub.checkpoint import save_checkpoint
from dlhub.config import append_jsonl, dataclass_to_dict, write_json
from dlhub.device import resolve_device
from dlhub.logging import get_logger
from dlhub.paths import build_run_paths
from dlhub.seed import set_seed

from .data import DataConfig, Vocab, get_dataloaders
from .model import ModelConfig, CompactBLIPModel, blip_lite_loss, caption_exact_match, token_accuracy


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 5
    learning_rate: float = 2e-3
    weight_decay: float = 1e-4
    seed: int = 42
    device: str = "auto"
    max_train_batches: int | None = None
    max_eval_batches: int | None = None
    run_name: str = "dev"
    hidden_dim: int = 64
    vision_width: int = 32
    embed_dim: int = 32
    itm_weight: float = 0.5


@dataclass(frozen=True)
class Stats:
    loss: float
    caption_loss: float
    itm_loss: float
    caption_token_acc: float
    caption_exact_match: float
    itm_acc: float


def parse_args() -> tuple[TrainConfig, DataConfig]:
    parser = argparse.ArgumentParser(
        description="Lesson 02 (Multimodal): BLIP-lite captioning plus image-text matching."
    )

    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=16)
    parser.add_argument("--max-text-length", type=int, default=10)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--negative-fraction", type=float, default=0.5)

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--run-name", type=str, default="dev")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)

    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--vision-width", type=int, default=32)
    parser.add_argument("--embed-dim", type=int, default=32)
    parser.add_argument("--itm-weight", type=float, default=0.5)

    args = parser.parse_args()

    train_cfg = TrainConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=args.device,
        max_train_batches=args.max_train_batches,
        max_eval_batches=args.max_eval_batches,
        run_name=args.run_name,
        hidden_dim=args.hidden_dim,
        vision_width=args.vision_width,
        embed_dim=args.embed_dim,
        itm_weight=args.itm_weight,
    )
    data_cfg = DataConfig(
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        image_size=args.image_size,
        max_text_length=args.max_text_length,
        val_fraction=args.val_fraction,
        seed=args.data_seed,
        num_workers=args.num_workers,
        negative_fraction=args.negative_fraction,
    )
    return train_cfg, data_cfg


def _move_batch(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    moved: dict[str, object] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def _run_epoch(
    *,
    model: CompactBLIPModel,
    loader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    pad_id: int,
    itm_weight: float,
    max_batches: int | None,
) -> Stats:
    is_train = optimizer is not None
    if is_train:
        model.train()
    else:
        model.eval()

    total_examples = 0
    total_loss = 0.0
    total_caption_loss = 0.0
    total_itm_loss = 0.0
    total_token_acc = 0.0
    total_exact = 0.0
    total_itm_acc = 0.0

    for step, batch in enumerate(loader):
        if max_batches is not None and step >= int(max_batches):
            break

        batch = _move_batch(batch, device)
        if is_train:
            optimizer.zero_grad(set_to_none=True)

        if is_train:
            outputs = model(batch)
            losses = blip_lite_loss(
                caption_logits=outputs["caption_logits"],
                itm_logits=outputs["itm_logits"],
                caption_targets=batch["caption_out_ids"],
                caption_mask=batch["caption_mask"],
                itm_targets=batch["itm_label"],
                pad_id=pad_id,
                itm_weight=itm_weight,
            )
        else:
            with torch.no_grad():
                outputs = model(batch)
                losses = blip_lite_loss(
                    caption_logits=outputs["caption_logits"],
                    itm_logits=outputs["itm_logits"],
                    caption_targets=batch["caption_out_ids"],
                    caption_mask=batch["caption_mask"],
                    itm_targets=batch["itm_label"],
                    pad_id=pad_id,
                    itm_weight=itm_weight,
                )

        if is_train:
            losses["loss"].backward()
            optimizer.step()

        batch_size = int(batch["image"].shape[0])
        total_examples += batch_size
        total_loss += float(losses["loss"].item()) * batch_size
        total_caption_loss += float(losses["caption_loss"].item()) * batch_size
        total_itm_loss += float(losses["itm_loss"].item()) * batch_size
        total_token_acc += token_accuracy(
            outputs["caption_logits"], batch["caption_out_ids"], batch["caption_mask"]
        ) * batch_size
        total_exact += caption_exact_match(
            outputs["caption_logits"], batch["caption_out_ids"], batch["caption_mask"]
        ) * batch_size
        itm_pred = outputs["itm_logits"].argmax(dim=-1)
        itm_acc = (itm_pred == batch["itm_label"]).to(torch.float32).mean().item()
        total_itm_acc += float(itm_acc) * batch_size

    if total_examples == 0:
        return Stats(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    return Stats(
        loss=total_loss / total_examples,
        caption_loss=total_caption_loss / total_examples,
        itm_loss=total_itm_loss / total_examples,
        caption_token_acc=total_token_acc / total_examples,
        caption_exact_match=total_exact / total_examples,
        itm_acc=total_itm_acc / total_examples,
    )


@torch.no_grad()
def _write_samples(
    *,
    model: CompactBLIPModel,
    loader,
    vocab: Vocab,
    device: torch.device,
    out_path,
    epoch: int,
) -> None:
    try:
        batch = next(iter(loader))
    except StopIteration:
        return

    moved = _move_batch(batch, device)
    outputs = model(moved)
    generated = model.greedy_generate(
        moved["image"], max_length=int(moved["caption_out_ids"].shape[1])
    ).cpu()
    itm_pred = outputs["itm_logits"].argmax(dim=-1).cpu().tolist()

    rows = []
    for idx in range(min(4, int(generated.shape[0]))):
        rows.append(
            {
                "epoch": int(epoch),
                "caption_gt": batch["caption_text"][idx],
                "caption_pred": " ".join(vocab.decode_ids(generated[idx].tolist())),
                "itm_text": batch["itm_text"][idx],
                "itm_label": int(batch["itm_label"][idx].item()),
                "itm_pred": int(itm_pred[idx]),
            }
        )

    with open(out_path, "a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_training(train_cfg: TrainConfig, data_cfg: DataConfig) -> int:
    set_seed(train_cfg.seed)
    device_info = resolve_device(train_cfg.device)

    paths = build_run_paths(
        track="multimodal",
        lesson="lesson_02_blip_compact_captioning",
        run_name=train_cfg.run_name,
    )
    logger = get_logger("multimodal.blip_compact", log_file=paths.logs_dir / "train.log")
    paths.run_dir.mkdir(parents=True, exist_ok=True)
    paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Device: %s (%s)", device_info.name, device_info.torch_device)
    logger.info("Outputs: %s", paths.run_dir)

    train_loader, val_loader, vocab = get_dataloaders(data_cfg)
    model = CompactBLIPModel(
        ModelConfig(
            vocab_size=vocab.size,
            pad_id=vocab.pad_id,
            bos_id=vocab.bos_id,
            eos_id=vocab.eos_id,
            max_text_length=int(data_cfg.max_text_length),
            hidden_dim=int(train_cfg.hidden_dim),
            vision_width=int(train_cfg.vision_width),
            embed_dim=int(train_cfg.embed_dim),
        )
    ).to(device_info.torch_device)

    write_json(
        paths.run_dir / "config.json",
        {
            "train": dataclass_to_dict(train_cfg),
            "data": dataclass_to_dict(data_cfg),
            "versions": {"python": sys.version, "torch": torch.__version__},
        },
    )
    write_json(paths.run_dir / "vocab.json", vocab.to_dict())

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.learning_rate),
        weight_decay=float(train_cfg.weight_decay),
    )

    metrics_path = paths.run_dir / "metrics.jsonl"
    samples_path = paths.run_dir / "samples.jsonl"
    for epoch in range(1, int(train_cfg.epochs) + 1):
        train_stats = _run_epoch(
            model=model,
            loader=train_loader,
            device=device_info.torch_device,
            optimizer=optimizer,
            pad_id=vocab.pad_id,
            itm_weight=float(train_cfg.itm_weight),
            max_batches=train_cfg.max_train_batches,
        )
        eval_stats = _run_epoch(
            model=model,
            loader=val_loader,
            device=device_info.torch_device,
            optimizer=None,
            pad_id=vocab.pad_id,
            itm_weight=float(train_cfg.itm_weight),
            max_batches=train_cfg.max_eval_batches,
        )

        _write_samples(
            model=model,
            loader=val_loader,
            vocab=vocab,
            device=device_info.torch_device,
            out_path=samples_path,
            epoch=epoch,
        )

        logger.info(
            "Epoch %d/%d | train loss %.4f cap %.4f itm %.4f tok %.3f em %.3f itm_acc %.3f | eval loss %.4f cap %.4f itm %.4f tok %.3f em %.3f itm_acc %.3f",
            epoch,
            train_cfg.epochs,
            train_stats.loss,
            train_stats.caption_loss,
            train_stats.itm_loss,
            train_stats.caption_token_acc,
            train_stats.caption_exact_match,
            train_stats.itm_acc,
            eval_stats.loss,
            eval_stats.caption_loss,
            eval_stats.itm_loss,
            eval_stats.caption_token_acc,
            eval_stats.caption_exact_match,
            eval_stats.itm_acc,
        )
        append_jsonl(
            metrics_path,
            {
                "epoch": epoch,
                "train_loss": train_stats.loss,
                "train_caption_loss": train_stats.caption_loss,
                "train_itm_loss": train_stats.itm_loss,
                "train_caption_token_acc": train_stats.caption_token_acc,
                "train_caption_exact_match": train_stats.caption_exact_match,
                "train_itm_acc": train_stats.itm_acc,
                "eval_loss": eval_stats.loss,
                "eval_caption_loss": eval_stats.caption_loss,
                "eval_itm_loss": eval_stats.itm_loss,
                "eval_caption_token_acc": eval_stats.caption_token_acc,
                "eval_caption_exact_match": eval_stats.caption_exact_match,
                "eval_itm_acc": eval_stats.itm_acc,
                "lr": optimizer.param_groups[0]["lr"],
            },
        )

    ckpt_path = save_checkpoint(
        paths.checkpoints_dir / "checkpoint.pt",
        model=model,
        optimizer=optimizer,
        epoch=int(train_cfg.epochs),
        extra={
            "track": "multimodal",
            "lesson": "lesson_02_blip_compact_captioning",
            "vocab_size": vocab.size,
        },
    )
    logger.info("Saved checkpoint to %s", ckpt_path)
    return 0


def main() -> int:
    if __package__ is None:
        raise RuntimeError(
            "Please run this lesson from the repo root as a module:\n"
            "  python -m tracks.multimodal.lesson_02_blip_compact_captioning.train"
        )

    train_cfg, data_cfg = parse_args()
    return run_training(train_cfg, data_cfg)


if __name__ == "__main__":
    raise SystemExit(main())
