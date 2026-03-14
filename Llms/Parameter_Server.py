from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ParameterServerConfig:
    num_workers: int
    consistency: str = "asp"
    staleness: int = 0


class ConsistencyController:
    def __init__(self, config: ParameterServerConfig) -> None:
        if int(config.num_workers) <= 0:
            raise ValueError("num_workers must be > 0")
        if str(config.consistency) not in {"asp", "bsp", "ssp"}:
            raise ValueError("consistency must be one of asp, bsp, or ssp")
        if int(config.staleness) < 0:
            raise ValueError("staleness must be >= 0")
        self.config = config
        self.clocks: dict[str, int] = {}

    def register(self, worker_id: str) -> None:
        if worker_id in self.clocks:
            return
        if len(self.clocks) >= int(self.config.num_workers):
            raise ValueError("cannot register more workers than configured")
        self.clocks[str(worker_id)] = 0

    def finish_step(self, worker_id: str) -> bool:
        if worker_id not in self.clocks:
            raise KeyError(f"unknown worker: {worker_id}")
        if str(self.config.consistency) == "asp":
            self.clocks[worker_id] += 1
            return True

        minimum_clock = min(self.clocks.values())
        staleness = 0 if str(self.config.consistency) == "bsp" else int(self.config.staleness)
        proposed = self.clocks[worker_id] + 1
        if proposed <= (minimum_clock + staleness + 1):
            self.clocks[worker_id] = proposed
            return True
        return False


class ParameterServer:
    def __init__(self, parameters: dict[str, torch.Tensor], config: ParameterServerConfig) -> None:
        self.parameters = {name: value.clone() for name, value in parameters.items()}
        self.config = config
        self.controller = ConsistencyController(config)

    def register_worker(self, worker_id: str) -> "ParameterWorker":
        self.controller.register(worker_id)
        return ParameterWorker(worker_id=str(worker_id), server=self)

    def pull(self, name: str, indices: torch.Tensor | None = None) -> torch.Tensor:
        tensor = self.parameters[str(name)]
        if indices is None:
            return tensor.clone()
        return tensor[indices.to(torch.long)].clone()

    def push(
        self,
        name: str,
        *,
        values: torch.Tensor,
        indices: torch.Tensor | None = None,
    ) -> None:
        tensor = self.parameters[str(name)]
        if indices is None:
            tensor.add_(values.to(dtype=tensor.dtype, device=tensor.device))
            return
        tensor[indices.to(torch.long)] += values.to(dtype=tensor.dtype, device=tensor.device)


@dataclass
class ParameterWorker:
    worker_id: str
    server: ParameterServer

    def pull(self, name: str, *, indices: torch.Tensor | None = None) -> torch.Tensor:
        return self.server.pull(name, indices=indices)

    def push(
        self,
        name: str,
        *,
        values: torch.Tensor,
        indices: torch.Tensor | None = None,
    ) -> None:
        self.server.push(name, values=values, indices=indices)

    def finish_step(self) -> bool:
        return self.server.controller.finish_step(self.worker_id)


__all__ = [
    "ConsistencyController",
    "ParameterServer",
    "ParameterServerConfig",
    "ParameterWorker",
]
