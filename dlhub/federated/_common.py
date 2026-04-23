from __future__ import annotations
from dataclasses import dataclass

import torch
import torch.nn.functional as F


def scale_hidden(base: int, width_mult: float, *, min_dim: int = 4, divisor: int = 4) -> int:
    dim = max(int(min_dim), int(round(float(base) * float(width_mult))))
    rem = dim % int(divisor)
    if rem:
        dim += int(divisor) - rem
    return dim


@dataclass(frozen=True)
class RoundState:
    server_params: torch.Tensor
    client_weights: torch.Tensor
    client_steps: torch.Tensor
    raw_updates: torch.Tensor
    client_repr: torch.Tensor
    global_repr: torch.Tensor


class FederatedStrategy:
    def __init__(
        self,
        *,
        family: str,
        param_dim: int,
        num_clients: int,
        local_steps: int,
        hidden_dim: int,
        step_scale: float,
    ) -> None:
        self.family = str(family)
        self.param_dim = int(param_dim)
        self.num_clients = int(num_clients)
        self.local_steps = int(local_steps)
        self.hidden_dim = int(hidden_dim)
        self.step_scale = float(step_scale)
        if self.param_dim <= 0:
            raise ValueError("param_dim must be > 0")
        if self.num_clients <= 0:
            raise ValueError("num_clients must be > 0")
        if self.local_steps <= 0:
            raise ValueError("local_steps must be > 0")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be > 0")

    def _sample_round_state(self, *, seed: int) -> RoundState:
        gen = torch.Generator().manual_seed(int(seed))
        base = torch.linspace(-0.25, 0.25, self.param_dim, dtype=torch.float32)
        server_params = base + 0.05 * torch.randn(self.param_dim, generator=gen)

        client_weights = torch.randint(8, 40, (self.num_clients,), generator=gen).to(torch.float32)
        client_steps = torch.randint(
            max(1, self.local_steps),
            self.local_steps + 3,
            (self.num_clients,),
            generator=gen,
        ).to(torch.float32)

        phase = torch.linspace(0.0, 3.14159, self.param_dim, dtype=torch.float32)
        harmonic = torch.sin(phase).unsqueeze(0)
        client_scale = torch.linspace(0.4, 1.2, self.num_clients, dtype=torch.float32).unsqueeze(1)
        noise = 0.05 * torch.randn(self.num_clients, self.param_dim, generator=gen)
        raw_updates = self.step_scale * (
            noise + client_scale * harmonic / client_steps.unsqueeze(1)
        )

        client_repr = F.normalize(
            torch.randn(self.num_clients, self.hidden_dim, generator=gen), dim=-1
        )
        global_repr = F.normalize(torch.randn(self.hidden_dim, generator=gen), dim=-1)

        return RoundState(
            server_params=server_params,
            client_weights=client_weights,
            client_steps=client_steps,
            raw_updates=raw_updates,
            client_repr=client_repr,
            global_repr=global_repr,
        )

    def simulate_round(self, *, seed: int = 0) -> dict[str, torch.Tensor]:
        raise NotImplementedError


def weighted_average(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    norm = weights / weights.sum().clamp_min(1e-8)
    return (norm.unsqueeze(1) * values).sum(dim=0)


def build_federated_strategy(
    strategy_cls,
    *,
    variants: dict[str, dict[str, float | int]],
    param_dim: int,
    num_clients: int,
    local_steps: int,
    variant: str,
    width_mult: float,
    family: str,
):
    name = str(variant).lower().strip()
    if name not in variants:
        raise ValueError(f"Unknown {family} variant: {variant!r}. Supported: {sorted(variants)}")
    spec = dict(variants[name])
    hidden = scale_hidden(int(spec.pop("hidden")), float(width_mult), min_dim=4, divisor=4)
    return strategy_cls(
        family=family,
        param_dim=int(param_dim),
        num_clients=int(num_clients),
        local_steps=int(local_steps),
        hidden_dim=int(hidden),
        **spec,
    )


def smoke_test_strategy(builder, variant: str) -> None:
    strategy = builder(param_dim=16, num_clients=4, local_steps=2, variant=variant, width_mult=0.5)
    out = strategy.simulate_round(seed=0)
    print(variant, {k: tuple(v.shape) for k, v in out.items() if torch.is_tensor(v)})
    assert "server_params" in out
    assert tuple(out["server_params"].shape) == (16,)
    print("ok")
