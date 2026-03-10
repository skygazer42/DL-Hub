from __future__ import annotations

import torch

from ._common import (
    FederatedStrategy,
    build_federated_strategy,
    smoke_test_strategy,
    weighted_average,
)


class IFCAStrategy(FederatedStrategy):
    def __init__(self, *, num_clusters: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.num_clusters = int(num_clusters)

    def simulate_round(self, *, seed: int = 0) -> dict[str, torch.Tensor]:
        state = self._sample_round_state(seed=seed)
        assignments = torch.arange(self.num_clients) % self.num_clusters
        cluster_params = []
        for cluster_id in range(self.num_clusters):
            mask = assignments == cluster_id
            cluster_params.append(
                weighted_average(state.raw_updates[mask], state.client_weights[mask])
            )
        cluster_params_t = torch.stack(cluster_params, dim=0)
        chosen = cluster_params_t[assignments]
        server_params = state.server_params + cluster_params_t.mean(dim=0)
        return {
            "server_params": server_params,
            "cluster_params": cluster_params_t,
            "cluster_assignments": assignments.to(torch.float32),
            "client_cluster_update": chosen,
        }


_VARIANTS: dict[str, dict[str, float | int]] = {
    "ifca_tiny": {"hidden": 8, "step_scale": 0.08, "num_clusters": 2},
    "ifca_small": {"hidden": 12, "step_scale": 0.12, "num_clusters": 3},
    "ifca_base": {"hidden": 16, "step_scale": 0.16, "num_clusters": 4},
}


def build_ifca_strategy(
    *,
    param_dim: int,
    num_clients: int,
    local_steps: int,
    variant: str = "ifca_tiny",
    width_mult: float = 1.0,
):
    return build_federated_strategy(
        IFCAStrategy,
        variants=_VARIANTS,
        param_dim=param_dim,
        num_clients=num_clients,
        local_steps=local_steps,
        variant=variant,
        width_mult=width_mult,
        family="ifca",
    )


if __name__ == "__main__":
    smoke_test_strategy(build_ifca_strategy, "ifca_tiny")
