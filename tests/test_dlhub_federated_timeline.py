import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_federated_timeline_metadata_covers_first_batch() -> None:
    from dlhub.federated._timeline import by_family, entries

    timeline = entries()
    assert len(timeline) >= 36

    groups = {entry.group for entry in timeline}
    assert groups == {
        "optimization",
        "control_variate",
        "personalized",
        "contrastive_personalized",
        "server_optimizer",
        "feature_normalization",
        "clustered_personalized",
        "representation_personalized",
        "prototype_personalized",
        "fairness",
        "long_tail_robustness",
        "split_learning",
        "heterogeneous_width",
        "distillation",
        "privacy",
        "compression",
        "secure_aggregation",
    }

    mapping = by_family()
    assert mapping["fedavg"].year == 2017
    assert mapping["fedprox"].group == "optimization"
    assert mapping["scaffold"].group == "control_variate"
    assert mapping["fednova"].year == 2020
    assert mapping["moon"].group == "contrastive_personalized"
    assert mapping["pfedme"].group == "personalized"
    assert mapping["feddyn"].group == "optimization"
    assert mapping["fedadam"].group == "server_optimizer"
    assert mapping["fedyogi"].group == "server_optimizer"
    assert mapping["fedbn"].group == "feature_normalization"
    assert mapping["ifca"].group == "clustered_personalized"
    assert mapping["ditto"].group == "personalized"
    assert mapping["fedper"].year == 2019
    assert mapping["per_fedavg"].group == "personalized"
    assert mapping["apfl"].group == "personalized"
    assert mapping["fedrep"].group == "representation_personalized"
    assert mapping["fedamp"].group == "personalized"
    assert mapping["fedproto"].group == "prototype_personalized"
    assert mapping["qfedavg"].group == "fairness"
    assert mapping["afl"].year == 2019
    assert mapping["term"].group == "fairness"
    assert mapping["fedrs"].group == "long_tail_robustness"
    assert mapping["fedlc"].group == "long_tail_robustness"
    assert mapping["fedrod"].group == "long_tail_robustness"
    assert mapping["splitfed"].group == "split_learning"
    assert mapping["splitfedv2"].group == "split_learning"
    assert mapping["heterofl"].group == "heterogeneous_width"
    assert mapping["fjord"].group == "heterogeneous_width"
    assert mapping["fedgkt"].group == "distillation"
    assert mapping["feddf"].group == "distillation"
    assert mapping["dp_fedavg"].group == "privacy"
    assert mapping["dp_fedprox"].group == "privacy"
    assert mapping["fedpaq"].group == "compression"
    assert mapping["stc"].group == "compression"
    assert mapping["secureagg"].group == "secure_aggregation"
    assert mapping["lightsecagg"].group == "secure_aggregation"


def test_federated_zoo_script_timeline() -> None:
    proc = subprocess.run(
        [sys.executable, "scripts/federated_zoo.py", "--timeline"],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "Federated learning timeline" in proc.stdout
    assert "total_families=" in proc.stdout
    assert "\n2017\n" in proc.stdout
    assert "fedavg [optimization]" in proc.stdout
    assert "scaffold [control_variate]" in proc.stdout
    assert "moon [contrastive_personalized]" in proc.stdout
    assert "fedadam [server_optimizer]" in proc.stdout
    assert "fedbn [feature_normalization]" in proc.stdout
    assert "fedrep [representation_personalized]" in proc.stdout
    assert "fedproto [prototype_personalized]" in proc.stdout
    assert "qfedavg [fairness]" in proc.stdout
    assert "fedrod [long_tail_robustness]" in proc.stdout
    assert "splitfed [split_learning]" in proc.stdout
    assert "heterofl [heterogeneous_width]" in proc.stdout
    assert "feddf [distillation]" in proc.stdout
    assert "dp_fedavg [privacy]" in proc.stdout
    assert "fedpaq [compression]" in proc.stdout
    assert "lightsecagg [secure_aggregation]" in proc.stdout
