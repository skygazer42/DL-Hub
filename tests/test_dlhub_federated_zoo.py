import pytest


torch = pytest.importorskip("torch")


def _sum_tensor_means(x):
    if torch.is_tensor(x):
        return x.to(torch.float32).mean()
    if isinstance(x, dict):
        return sum((_sum_tensor_means(v) for v in x.values()), start=torch.tensor(0.0))
    if isinstance(x, (list, tuple)):
        return sum((_sum_tensor_means(v) for v in x), start=torch.tensor(0.0))
    raise TypeError(f"Unsupported output type in federated zoo smoke: {type(x)!r}")


def test_federated_zoo_lists_first_batch_arches() -> None:
    from dlhub.federated_zoo import list_local_arches

    arches = list_local_arches()
    assert len(arches) >= 108
    assert "dlfed:fedavg_tiny" in arches
    assert "dlfed:fedprox_tiny" in arches
    assert "dlfed:scaffold_tiny" in arches
    assert "dlfed:fednova_tiny" in arches
    assert "dlfed:moon_tiny" in arches
    assert "dlfed:pfedme_tiny" in arches
    assert "dlfed:feddyn_tiny" in arches
    assert "dlfed:fedadam_tiny" in arches
    assert "dlfed:fedyogi_tiny" in arches
    assert "dlfed:fedbn_tiny" in arches
    assert "dlfed:ifca_tiny" in arches
    assert "dlfed:ditto_tiny" in arches
    assert "dlfed:fedper_tiny" in arches
    assert "dlfed:per_fedavg_tiny" in arches
    assert "dlfed:apfl_tiny" in arches
    assert "dlfed:fedrep_tiny" in arches
    assert "dlfed:fedamp_tiny" in arches
    assert "dlfed:fedproto_tiny" in arches
    assert "dlfed:qfedavg_tiny" in arches
    assert "dlfed:afl_tiny" in arches
    assert "dlfed:term_tiny" in arches
    assert "dlfed:fedrs_tiny" in arches
    assert "dlfed:fedlc_tiny" in arches
    assert "dlfed:fedrod_tiny" in arches
    assert "dlfed:splitfed_tiny" in arches
    assert "dlfed:splitfedv2_tiny" in arches
    assert "dlfed:heterofl_tiny" in arches
    assert "dlfed:fjord_tiny" in arches
    assert "dlfed:fedgkt_tiny" in arches
    assert "dlfed:feddf_tiny" in arches
    assert "dlfed:dp_fedavg_tiny" in arches
    assert "dlfed:dp_fedprox_tiny" in arches
    assert "dlfed:fedpaq_tiny" in arches
    assert "dlfed:stc_tiny" in arches
    assert "dlfed:secureagg_tiny" in arches
    assert "dlfed:lightsecagg_tiny" in arches


@pytest.mark.parametrize(
    "arch_id",
    [
        "dlfed:fedavg_tiny",
        "dlfed:fedprox_tiny",
        "dlfed:scaffold_tiny",
        "dlfed:moon_tiny",
        "dlfed:fedadam_tiny",
        "dlfed:fedbn_tiny",
        "dlfed:ifca_tiny",
        "dlfed:ditto_tiny",
        "dlfed:fedper_tiny",
        "dlfed:fedrep_tiny",
        "dlfed:fedproto_tiny",
        "dlfed:qfedavg_tiny",
        "dlfed:term_tiny",
        "dlfed:fedrs_tiny",
        "dlfed:fedrod_tiny",
        "dlfed:splitfed_tiny",
        "dlfed:heterofl_tiny",
        "dlfed:fedgkt_tiny",
        "dlfed:feddf_tiny",
        "dlfed:dp_fedavg_tiny",
        "dlfed:fedpaq_tiny",
        "dlfed:secureagg_tiny",
        "dlfed:lightsecagg_tiny",
    ],
)
def test_federated_zoo_build_and_simulate_smoke(arch_id: str) -> None:
    from dlhub.federated_zoo import build_local_strategy

    strategy = build_local_strategy(
        arch_id,
        param_dim=16,
        num_clients=4,
        local_steps=2,
        width_mult=0.5,
    )
    out = strategy.simulate_round(seed=0)
    assert isinstance(out, dict)
    assert "server_params" in out
    loss = _sum_tensor_means(out)
    assert torch.isfinite(loss)
