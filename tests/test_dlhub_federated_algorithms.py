import pytest


torch = pytest.importorskip("torch")


def _sum_tensor_means(x):
    if torch.is_tensor(x):
        return x.to(torch.float32).mean()
    if isinstance(x, dict):
        return sum((_sum_tensor_means(v) for v in x.values()), start=torch.tensor(0.0))
    if isinstance(x, (list, tuple)):
        return sum((_sum_tensor_means(v) for v in x), start=torch.tensor(0.0))
    raise TypeError(f"Unsupported output type in federated smoke: {type(x)!r}")


@pytest.mark.parametrize(
    "builder_name,kwargs",
    [
        ("build_fedavg_strategy", {"param_dim": 16, "num_clients": 4, "local_steps": 2, "variant": "fedavg_tiny", "width_mult": 0.5}),
        ("build_fedprox_strategy", {"param_dim": 16, "num_clients": 4, "local_steps": 2, "variant": "fedprox_tiny", "width_mult": 0.5}),
        ("build_scaffold_strategy", {"param_dim": 16, "num_clients": 4, "local_steps": 2, "variant": "scaffold_tiny", "width_mult": 0.5}),
        ("build_fednova_strategy", {"param_dim": 16, "num_clients": 4, "local_steps": 2, "variant": "fednova_tiny", "width_mult": 0.5}),
        ("build_moon_strategy", {"param_dim": 16, "num_clients": 4, "local_steps": 2, "variant": "moon_tiny", "width_mult": 0.5}),
        ("build_pfedme_strategy", {"param_dim": 16, "num_clients": 4, "local_steps": 2, "variant": "pfedme_tiny", "width_mult": 0.5}),
        ("build_feddyn_strategy", {"param_dim": 16, "num_clients": 4, "local_steps": 2, "variant": "feddyn_tiny", "width_mult": 0.5}),
        ("build_fedadam_strategy", {"param_dim": 16, "num_clients": 4, "local_steps": 2, "variant": "fedadam_tiny", "width_mult": 0.5}),
        ("build_fedyogi_strategy", {"param_dim": 16, "num_clients": 4, "local_steps": 2, "variant": "fedyogi_tiny", "width_mult": 0.5}),
        ("build_fedbn_strategy", {"param_dim": 16, "num_clients": 4, "local_steps": 2, "variant": "fedbn_tiny", "width_mult": 0.5}),
        ("build_ifca_strategy", {"param_dim": 16, "num_clients": 4, "local_steps": 2, "variant": "ifca_tiny", "width_mult": 0.5}),
        ("build_ditto_strategy", {"param_dim": 16, "num_clients": 4, "local_steps": 2, "variant": "ditto_tiny", "width_mult": 0.5}),
        ("build_fedper_strategy", {"param_dim": 16, "num_clients": 4, "local_steps": 2, "variant": "fedper_tiny", "width_mult": 0.5}),
        ("build_per_fedavg_strategy", {"param_dim": 16, "num_clients": 4, "local_steps": 2, "variant": "per_fedavg_tiny", "width_mult": 0.5}),
        ("build_apfl_strategy", {"param_dim": 16, "num_clients": 4, "local_steps": 2, "variant": "apfl_tiny", "width_mult": 0.5}),
        ("build_fedrep_strategy", {"param_dim": 16, "num_clients": 4, "local_steps": 2, "variant": "fedrep_tiny", "width_mult": 0.5}),
        ("build_fedamp_strategy", {"param_dim": 16, "num_clients": 4, "local_steps": 2, "variant": "fedamp_tiny", "width_mult": 0.5}),
        ("build_fedproto_strategy", {"param_dim": 16, "num_clients": 4, "local_steps": 2, "variant": "fedproto_tiny", "width_mult": 0.5}),
        ("build_qfedavg_strategy", {"param_dim": 16, "num_clients": 4, "local_steps": 2, "variant": "qfedavg_tiny", "width_mult": 0.5}),
        ("build_afl_strategy", {"param_dim": 16, "num_clients": 4, "local_steps": 2, "variant": "afl_tiny", "width_mult": 0.5}),
        ("build_term_strategy", {"param_dim": 16, "num_clients": 4, "local_steps": 2, "variant": "term_tiny", "width_mult": 0.5}),
        ("build_fedrs_strategy", {"param_dim": 16, "num_clients": 4, "local_steps": 2, "variant": "fedrs_tiny", "width_mult": 0.5}),
        ("build_fedlc_strategy", {"param_dim": 16, "num_clients": 4, "local_steps": 2, "variant": "fedlc_tiny", "width_mult": 0.5}),
        ("build_fedrod_strategy", {"param_dim": 16, "num_clients": 4, "local_steps": 2, "variant": "fedrod_tiny", "width_mult": 0.5}),
        ("build_splitfed_strategy", {"param_dim": 16, "num_clients": 4, "local_steps": 2, "variant": "splitfed_tiny", "width_mult": 0.5}),
        ("build_splitfedv2_strategy", {"param_dim": 16, "num_clients": 4, "local_steps": 2, "variant": "splitfedv2_tiny", "width_mult": 0.5}),
        ("build_heterofl_strategy", {"param_dim": 16, "num_clients": 4, "local_steps": 2, "variant": "heterofl_tiny", "width_mult": 0.5}),
        ("build_fjord_strategy", {"param_dim": 16, "num_clients": 4, "local_steps": 2, "variant": "fjord_tiny", "width_mult": 0.5}),
        ("build_fedgkt_strategy", {"param_dim": 16, "num_clients": 4, "local_steps": 2, "variant": "fedgkt_tiny", "width_mult": 0.5}),
        ("build_feddf_strategy", {"param_dim": 16, "num_clients": 4, "local_steps": 2, "variant": "feddf_tiny", "width_mult": 0.5}),
        ("build_dp_fedavg_strategy", {"param_dim": 16, "num_clients": 4, "local_steps": 2, "variant": "dp_fedavg_tiny", "width_mult": 0.5}),
        ("build_dp_fedprox_strategy", {"param_dim": 16, "num_clients": 4, "local_steps": 2, "variant": "dp_fedprox_tiny", "width_mult": 0.5}),
        ("build_fedpaq_strategy", {"param_dim": 16, "num_clients": 4, "local_steps": 2, "variant": "fedpaq_tiny", "width_mult": 0.5}),
        ("build_stc_strategy", {"param_dim": 16, "num_clients": 4, "local_steps": 2, "variant": "stc_tiny", "width_mult": 0.5}),
        ("build_secureagg_strategy", {"param_dim": 16, "num_clients": 4, "local_steps": 2, "variant": "secureagg_tiny", "width_mult": 0.5}),
        ("build_lightsecagg_strategy", {"param_dim": 16, "num_clients": 4, "local_steps": 2, "variant": "lightsecagg_tiny", "width_mult": 0.5}),
    ],
)
def test_federated_strategies_simulate_round_smoke(builder_name: str, kwargs: dict) -> None:
    import dlhub.federated as fed

    build = getattr(fed, builder_name)
    strategy = build(**kwargs)
    out = strategy.simulate_round(seed=0)
    assert isinstance(out, dict)
    assert "server_params" in out
    assert tuple(out["server_params"].shape) == (int(kwargs["param_dim"]),)
    loss = _sum_tensor_means(out)
    assert torch.isfinite(loss)
