from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ZeROConfig:
    stage: int
    world_size: int = 1
    rank: int = 0


@dataclass(frozen=True)
class ZeROPartitionPlan:
    stage: int
    world_size: int
    rank: int
    shard_start: int
    shard_end: int
    partitions_optimizer_state: bool
    partitions_gradients: bool
    partitions_parameters: bool


@dataclass
class ZeROStateShard:
    engine: "ZeROEngine"
    plan: ZeROPartitionPlan
    parameter: torch.Tensor
    gradient: torch.Tensor
    optimizer_state: dict[str, torch.Tensor]


class ZeROPartitioner:
    def __init__(self, config: ZeROConfig) -> None:
        self.config = config
        self._validate()

    def _validate(self) -> None:
        if int(self.config.stage) not in (1, 2, 3):
            raise ValueError("stage must be one of 1, 2, or 3")
        if int(self.config.world_size) <= 0:
            raise ValueError("world_size must be > 0")
        if int(self.config.rank) < 0 or int(self.config.rank) >= int(self.config.world_size):
            raise ValueError("rank must be within the data parallel group")

    def plan(self, numel: int) -> ZeROPartitionPlan:
        if int(numel) <= 0:
            raise ValueError("numel must be > 0")
        shard_size = (int(numel) + int(self.config.world_size) - 1) // int(self.config.world_size)
        start = int(self.config.rank) * shard_size
        end = min(start + shard_size, int(numel))
        return ZeROPartitionPlan(
            stage=int(self.config.stage),
            world_size=int(self.config.world_size),
            rank=int(self.config.rank),
            shard_start=start,
            shard_end=end,
            partitions_optimizer_state=True,
            partitions_gradients=int(self.config.stage) >= 2,
            partitions_parameters=int(self.config.stage) >= 3,
        )

    def shard_tensor(self, tensor: torch.Tensor, *, partitioned: bool) -> torch.Tensor:
        flat = tensor.reshape(-1)
        if not partitioned:
            return flat.clone()
        plan = self.plan(flat.numel())
        return flat[plan.shard_start : plan.shard_end].clone()

    def gather(self, shards: list[torch.Tensor] | tuple[torch.Tensor, ...]) -> torch.Tensor:
        if not shards:
            raise ValueError("shards cannot be empty")
        return torch.cat([shard.reshape(-1) for shard in shards], dim=0)


class ZeROEngine:
    def __init__(self, config: ZeROConfig) -> None:
        self.config = config
        self.partitioner = ZeROPartitioner(config)

    def partition_states(
        self,
        *,
        parameter: torch.Tensor,
        gradient: torch.Tensor,
        optimizer_state: dict[str, torch.Tensor],
    ) -> ZeROStateShard:
        plan = self.partitioner.plan(parameter.numel())
        parameter_shard = self.partitioner.shard_tensor(
            parameter,
            partitioned=plan.partitions_parameters,
        )
        gradient_shard = self.partitioner.shard_tensor(
            gradient,
            partitioned=plan.partitions_gradients,
        )
        optimizer_shards = {
            name: self.partitioner.shard_tensor(value, partitioned=plan.partitions_optimizer_state)
            for name, value in optimizer_state.items()
        }
        return ZeROStateShard(
            engine=self,
            plan=plan,
            parameter=parameter_shard,
            gradient=gradient_shard,
            optimizer_state=optimizer_shards,
        )

    def gather_parameters(self, shards: list[torch.Tensor] | tuple[torch.Tensor, ...]) -> torch.Tensor:
        return self.partitioner.gather(shards)


__all__ = [
    "ZeROConfig",
    "ZeROEngine",
    "ZeROPartitioner",
    "ZeROPartitionPlan",
    "ZeROStateShard",
]
