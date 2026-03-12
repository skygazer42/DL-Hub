from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


def _balanced_partition_sizes(total_layers: int, num_partitions: int) -> list[int]:
    if int(total_layers) <= 0:
        raise ValueError("total_layers must be > 0")
    if int(num_partitions) <= 0:
        raise ValueError("num_partitions must be > 0")
    if int(num_partitions) > int(total_layers):
        raise ValueError("num_partitions cannot exceed the number of layers")
    base, remainder = divmod(int(total_layers), int(num_partitions))
    return [base + (1 if index < remainder else 0) for index in range(int(num_partitions))]


@dataclass(frozen=True)
class GPipeConfig:
    num_partitions: int
    micro_batches: int = 1
    rematerialization: bool = True
    split_dim: int = 0


class PartitionedCell(nn.Module):
    def __init__(self, layers: list[nn.Module]) -> None:
        super().__init__()
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class GPipeSequential(nn.Module):
    def __init__(self, layers: nn.ModuleList | list[nn.Module] | tuple[nn.Module, ...], config: GPipeConfig) -> None:
        super().__init__()
        modules = list(layers)
        if int(config.micro_batches) <= 0:
            raise ValueError("micro_batches must be > 0")
        self.config = config
        self.rematerialization = bool(config.rematerialization)
        self.partition_sizes = _balanced_partition_sizes(len(modules), int(config.num_partitions))
        self.bubble_steps = int(config.num_partitions) - 1

        cells: list[PartitionedCell] = []
        start = 0
        for size in self.partition_sizes:
            end = start + size
            cells.append(PartitionedCell(modules[start:end]))
            start = end
        self.cells = nn.ModuleList(cells)

    def pipeline_schedule(self) -> tuple[tuple[tuple[int, int], ...], ...]:
        num_partitions = len(self.cells)
        num_micro_batches = int(self.config.micro_batches)
        schedule: list[tuple[tuple[int, int], ...]] = []
        for clock in range(num_micro_batches + num_partitions - 1):
            active: list[tuple[int, int]] = []
            for partition_index in range(num_partitions):
                micro_batch_index = clock - partition_index
                if 0 <= micro_batch_index < num_micro_batches:
                    active.append((partition_index, micro_batch_index))
            schedule.append(tuple(active))
        return tuple(schedule)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        split_dim = int(self.config.split_dim)
        if x.shape[split_dim] < int(self.config.micro_batches):
            raise ValueError("micro_batches cannot exceed the split dimension size")

        outputs = []
        for micro_batch in torch.tensor_split(x, int(self.config.micro_batches), dim=split_dim):
            hidden = micro_batch
            for cell in self.cells:
                hidden = cell(hidden)
            outputs.append(hidden)
        return torch.cat(outputs, dim=split_dim)


__all__ = [
    "GPipeConfig",
    "GPipeSequential",
    "PartitionedCell",
]
