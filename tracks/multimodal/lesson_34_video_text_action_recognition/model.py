from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


class TemporalVideoEncoder(nn.Module):
    def __init__(self, *, feature_dim: int, hidden_dim: int, num_frames: int) -> None:
        super().__init__()
        self.input_proj = nn.Linear(int(feature_dim), int(hidden_dim))
        self.pos_embed = nn.Parameter(torch.zeros(int(num_frames), int(hidden_dim)))
        self.temporal_rnn = nn.GRU(int(hidden_dim), int(hidden_dim), batch_first=True)
        self.norm = nn.LayerNorm(int(hidden_dim))

    def forward(self, video_features: torch.Tensor) -> torch.Tensor:
        feat = self.input_proj(video_features.to(torch.float32))
        seq_len = int(feat.shape[1])
        feat = feat + self.pos_embed[:seq_len].unsqueeze(0)
        out, _ = self.temporal_rnn(feat)
        return self.norm(out)


class QueryEncoder(nn.Module):
    def __init__(self, *, vocab_size: int, pad_id: int, text_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(int(vocab_size), int(text_dim), padding_idx=int(pad_id))
        self.query_rnn = nn.GRU(int(text_dim), int(hidden_dim), batch_first=True)
        self.norm = nn.LayerNorm(int(hidden_dim))

    def forward(self, query_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        del attention_mask
        emb = self.embedding(query_ids.to(torch.long))
        _out, hidden = self.query_rnn(emb)
        return self.norm(hidden[-1])


@dataclass(frozen=True)
class ActionRecognitionModelConfig:
    vocab_size: int
    pad_id: int
    num_frames: int
    feature_dim: int = 24
    hidden_dim: int = 64
    text_dim: int = 32
    num_classes: int = 3


class ToyActionRecognitionModel(nn.Module):
    def __init__(self, cfg: ActionRecognitionModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.video_encoder = TemporalVideoEncoder(
            feature_dim=int(cfg.feature_dim),
            hidden_dim=int(cfg.hidden_dim),
            num_frames=int(cfg.num_frames),
        )
        self.query_encoder = QueryEncoder(
            vocab_size=int(cfg.vocab_size),
            pad_id=int(cfg.pad_id),
            text_dim=int(cfg.text_dim),
            hidden_dim=int(cfg.hidden_dim),
        )
        self.fusion = nn.Sequential(
            nn.Linear(int(cfg.hidden_dim) * 2, int(cfg.hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(cfg.hidden_dim), int(cfg.hidden_dim)),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(int(cfg.hidden_dim), int(cfg.num_classes))

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        video_feat = self.video_encoder(batch["video_features"])
        pooled_video = video_feat.mean(dim=1)
        query_feat = self.query_encoder(batch["query_ids"], batch["attention_mask"])
        fused = self.fusion(torch.cat([pooled_video, query_feat], dim=-1))
        logits = self.classifier(fused)
        pred_labels = logits.argmax(dim=1)
        return {
            "logits": logits,
            "pred_labels": pred_labels,
        }


def action_recognition_loss(*, logits: torch.Tensor, labels: torch.Tensor) -> dict[str, torch.Tensor]:
    cls_loss = F.cross_entropy(logits, labels.to(torch.long))
    return {"loss": cls_loss, "cls_loss": cls_loss}


@torch.no_grad()
def classification_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    target = labels.to(torch.long)
    return float((pred == target).to(torch.float32).mean().item())


__all__ = [
    "ActionRecognitionModelConfig",
    "QueryEncoder",
    "TemporalVideoEncoder",
    "ToyActionRecognitionModel",
    "action_recognition_loss",
    "classification_accuracy",
]
