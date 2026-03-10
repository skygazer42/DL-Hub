from dataclasses import dataclass


@dataclass(frozen=True)
class DataConfig:
    num_samples: int = 1024


def get_dataloaders(cfg: DataConfig):
    raise NotImplementedError


__all__ = ["DataConfig", "get_dataloaders"]

