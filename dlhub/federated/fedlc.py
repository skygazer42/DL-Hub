from __future__ import annotations

import torch

from ._common import FederatedStrategy, build_federated_strategy, smoke_test_strategy


class FedLCStrategy(FederatedStrategy):
    def __init__(self, *, calibration_strength: float, **kwargs) -> None:
        super().__init__(**kwargs)
        self.calibration_strength = float(calibration_strength)

    def simulate_round(self, *, seed: int = 0) -> dict[str, torch.Tensor]:
        state = self._sample_round_state(seed=seed)
        logits = state.client_repr[:, : min(4, self.hidden_dim)]
        calibrated = logits - self.calibration_strength * logits.mean(dim=0, keepdim=True)
        correction = calibrated.mean() * 0.05
        server_params = state.server_params + state.raw_updates.mean(dim=0) + correction
        return {
            "server_params": server_params,
            "calibrated_logits": calibrated,
            "calibration_bias": calibrated.mean(dim=0),
        }


_VARIANTS: dict[str, dict[str, float | int]] = {
    "fedlc_tiny": {"hidden": 8, "step_scale": 0.08, "calibration_strength": 0.3},
    "fedlc_small": {"hidden": 12, "step_scale": 0.12, "calibration_strength": 0.45},
    "fedlc_base": {"hidden": 16, "step_scale": 0.16, "calibration_strength": 0.6},
}


def build_fedlc_strategy(
    *,
    param_dim: int,
    num_clients: int,
    local_steps: int,
    variant: str = "fedlc_tiny",
    width_mult: float = 1.0,
):
    return build_federated_strategy(
        FedLCStrategy,
        variants=_VARIANTS,
        param_dim=param_dim,
        num_clients=num_clients,
        local_steps=local_steps,
        variant=variant,
        width_mult=width_mult,
        family="fedlc",
    )


if __name__ == "__main__":
    smoke_test_strategy(build_fedlc_strategy, "fedlc_tiny")
