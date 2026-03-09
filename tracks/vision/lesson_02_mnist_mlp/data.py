
from dataclasses import dataclass

from torch.utils.data import DataLoader

from dlhub.data.mnist import get_mnist_dataloaders


@dataclass(frozen=True)
class DataConfig:
    dataset: str = "mnist"  # mnist | fake
    data_dir: str = ".data"
    batch_size: int = 128
    num_workers: int = 2


def get_dataloaders(config: DataConfig) -> tuple[DataLoader, DataLoader]:
    return get_mnist_dataloaders(
        dataset=config.dataset,
        data_dir=config.data_dir,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )
