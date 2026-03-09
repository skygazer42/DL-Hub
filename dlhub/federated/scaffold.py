from __future__ import annotations

import torch

from ._common import FederatedStrategy, build_federated_strategy, smoke_test_strategy, weighted_average


class ScaffoldStrategy(FederatedStrategy):
    def __init__(self, *, control_scale: float, **kwargs) -> None:
        super().__init__(**kwargs)
        self.control_scale = float(control_scale)

    def simulate_round(self, *, seed: int = 0) -> dict[str, torch.Tensor]:
        state = self._sample_round_state(seed=seed)
        gen = torch.Generator().manual_seed(int(seed) + 17)
        server_control = 0.05 * torch.randn(self.param_dim, generator=gen)
        client_controls = 0.05 * torch.randn(self.num_clients, self.param_dim, generator=gen)
        corrected = state.raw_updates + self.control_scale * (server_control.unsqueeze(0) - client_controls)
        client_params = state.server_params.unsqueeze(0) + corrected
        server_params = weighted_average(client_params, state.client_weights)
        new_server_control = server_control + corrected.mean(dim=0) * 0.1
        return {
            "server_params": server_params,
            "client_params": client_params,
            "server_control": new_server_control,
            "client_controls": client_controls,
        }


_VARIANTS: dict[str, dict[str, float | int]] = {
    "scaffold_tiny": {"hidden": 8, "step_scale": 0.08, "control_scale": 0.25},
    "scaffold_small": {"hidden": 12, "step_scale": 0.12, "control_scale": 0.35},
    "scaffold_base": {"hidden": 16, "step_scale": 0.16, "control_scale": 0.5},
}


def build_scaffold_strategy(
    *,
    param_dim: int,
    num_clients: int,
    local_steps: int,
    variant: str = "scaffold_tiny",
    width_mult: float = 1.0,
):
    return build_federated_strategy(
        ScaffoldStrategy,
        variants=_VARIANTS,
        param_dim=param_dim,
        num_clients=num_clients,
        local_steps=local_steps,
        variant=variant,
        width_mult=width_mult,
        family="scaffold",
    )


if __name__ == "__main__":
    smoke_test_strategy(build_scaffold_strategy, "scaffold_tiny")
