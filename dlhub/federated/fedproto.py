from __future__ import annotations

import torch

from ._common import FederatedStrategy, build_federated_strategy, smoke_test_strategy


class FedProtoStrategy(FederatedStrategy):
    def __init__(self, *, num_prototypes: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.num_prototypes = int(num_prototypes)

    def simulate_round(self, *, seed: int = 0) -> dict[str, torch.Tensor]:
        state = self._sample_round_state(seed=seed)
        client_proto = state.client_repr[:, : self.num_prototypes]
        global_proto = client_proto.mean(dim=0)
        proto_bias = global_proto.mean() * 0.02
        server_params = state.server_params + state.raw_updates.mean(dim=0) + proto_bias
        return {
            "server_params": server_params,
            "client_prototypes": client_proto,
            "global_prototype": global_proto,
        }


_VARIANTS: dict[str, dict[str, float | int]] = {
    "fedproto_tiny": {"hidden": 8, "step_scale": 0.08, "num_prototypes": 4},
    "fedproto_small": {"hidden": 12, "step_scale": 0.12, "num_prototypes": 6},
    "fedproto_base": {"hidden": 16, "step_scale": 0.16, "num_prototypes": 8},
}


def build_fedproto_strategy(
    *,
    param_dim: int,
    num_clients: int,
    local_steps: int,
    variant: str = "fedproto_tiny",
    width_mult: float = 1.0,
):
    return build_federated_strategy(
        FedProtoStrategy,
        variants=_VARIANTS,
        param_dim=param_dim,
        num_clients=num_clients,
        local_steps=local_steps,
        variant=variant,
        width_mult=width_mult,
        family="fedproto",
    )


if __name__ == "__main__":
    smoke_test_strategy(build_fedproto_strategy, "fedproto_tiny")
