from __future__ import annotations

import torch

from ._common import FederatedStrategy, build_federated_strategy, smoke_test_strategy


class FedDFStrategy(FederatedStrategy):
    def __init__(self, *, distill_temp: float, **kwargs) -> None:
        super().__init__(**kwargs)
        self.distill_temp = float(distill_temp)

    def simulate_round(self, *, seed: int = 0) -> dict[str, torch.Tensor]:
        state = self._sample_round_state(seed=seed)
        ensemble_logits = state.client_repr / max(self.distill_temp, 1e-6)
        teacher = torch.softmax(ensemble_logits, dim=-1).mean(dim=0)
        distilled = teacher.mean() * 0.05
        server_params = state.server_params + state.raw_updates.mean(dim=0) + distilled
        return {
            "server_params": server_params,
            "ensemble_teacher": teacher,
            "ensemble_logits": ensemble_logits,
        }


_VARIANTS: dict[str, dict[str, float | int]] = {
    "feddf_tiny": {"hidden": 8, "step_scale": 0.08, "distill_temp": 1.0},
    "feddf_small": {"hidden": 12, "step_scale": 0.12, "distill_temp": 0.7},
    "feddf_base": {"hidden": 16, "step_scale": 0.16, "distill_temp": 0.5},
}


def build_feddf_strategy(
    *,
    param_dim: int,
    num_clients: int,
    local_steps: int,
    variant: str = "feddf_tiny",
    width_mult: float = 1.0,
):
    return build_federated_strategy(
        FedDFStrategy,
        variants=_VARIANTS,
        param_dim=param_dim,
        num_clients=num_clients,
        local_steps=local_steps,
        variant=variant,
        width_mult=width_mult,
        family="feddf",
    )


if __name__ == "__main__":
    smoke_test_strategy(build_feddf_strategy, "feddf_tiny")
