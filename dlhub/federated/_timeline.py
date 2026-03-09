"""Federated learning timeline metadata (best-effort, for docs/CLI)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TimelineEntry:
    year: int | None
    family: str
    method: str
    group: str
    reference: str | None = None


_ENTRIES: list[TimelineEntry] = [
    TimelineEntry(2017, "fedavg", "FedAvg (iterative model averaging)", "optimization"),
    TimelineEntry(2020, "fedprox", "FedProx (proximal regularized FedAvg)", "optimization"),
    TimelineEntry(2020, "scaffold", "SCAFFOLD (control variates for client drift)", "control_variate"),
    TimelineEntry(2020, "fednova", "FedNova (normalized averaging)", "optimization"),
    TimelineEntry(2021, "feddyn", "FedDyn (dynamic regularization for federated optimization)", "optimization"),
    TimelineEntry(2021, "fedadam", "FedAdam (server-side Adam in federated optimization)", "server_optimizer"),
    TimelineEntry(2021, "fedyogi", "FedYogi (server-side Yogi in federated optimization)", "server_optimizer"),
    TimelineEntry(2021, "fedbn", "FedBN (local batch normalization for feature-shift non-IID clients)", "feature_normalization"),
    TimelineEntry(2021, "ifca", "IFCA (clustered personalized federated learning)", "clustered_personalized"),
    TimelineEntry(2020, "pfedme", "pFedMe (personalized meta-regularized federated learning)", "personalized"),
    TimelineEntry(2021, "moon", "MOON (model-contrastive personalized federated learning)", "contrastive_personalized"),
    TimelineEntry(2021, "ditto", "Ditto (personalized federated optimization with proximal local heads)", "personalized"),
    TimelineEntry(2019, "fedper", "FedPer (base/head split personalization)", "personalized"),
    TimelineEntry(2020, "per_fedavg", "Per-FedAvg (meta-learning personalized federated averaging)", "personalized"),
    TimelineEntry(2020, "apfl", "APFL (adaptive personalized federated learning)", "personalized"),
    TimelineEntry(2021, "fedrep", "FedRep (shared representation with personalized heads)", "representation_personalized"),
    TimelineEntry(2020, "fedamp", "FedAMP (attentive message passing personalization)", "personalized"),
    TimelineEntry(2022, "fedproto", "FedProto (prototype-based federated learning)", "prototype_personalized"),
    TimelineEntry(2020, "qfedavg", "q-FedAvg (fair resource allocation in federated optimization)", "fairness"),
    TimelineEntry(2019, "afl", "AFL (agnostic federated learning)", "fairness"),
    TimelineEntry(2023, "term", "TERM (tilted empirical risk minimization in federated learning)", "fairness"),
    TimelineEntry(2023, "fedrs", "FedRS (re-balancing softmax for class-imbalanced federated learning)", "long_tail_robustness"),
    TimelineEntry(2022, "fedlc", "FedLC (logit calibration for long-tailed federated learning)", "long_tail_robustness"),
    TimelineEntry(2023, "fedrod", "FedRoD (robust distillation for long-tailed federated learning)", "long_tail_robustness"),
    TimelineEntry(2020, "splitfed", "SplitFed (federated split learning)", "split_learning"),
    TimelineEntry(2021, "splitfedv2", "SplitFedV2 (enhanced split-federated hybrid training)", "split_learning"),
    TimelineEntry(2021, "heterofl", "HeteroFL (heterogeneous-width federated learning)", "heterogeneous_width"),
    TimelineEntry(2021, "fjord", "FjORD (federated dropout for heterogeneous clients)", "heterogeneous_width"),
    TimelineEntry(2020, "fedgkt", "FedGKT (federated group knowledge transfer)", "distillation"),
    TimelineEntry(2020, "feddf", "FedDF (ensemble distillation for federated learning)", "distillation"),
    TimelineEntry(2019, "dp_fedavg", "DP-FedAvg (differentially private federated averaging)", "privacy"),
    TimelineEntry(2020, "dp_fedprox", "DP-FedProx (differentially private proximal federated learning)", "privacy"),
    TimelineEntry(2020, "fedpaq", "FedPAQ (periodic averaging and quantization)", "compression"),
    TimelineEntry(2019, "stc", "STC (sparse ternary compression)", "compression"),
    TimelineEntry(2017, "secureagg", "Secure Aggregation (privacy-preserving secure summation)", "secure_aggregation"),
    TimelineEntry(2022, "lightsecagg", "LightSecAgg (lightweight secure aggregation)", "secure_aggregation"),
]


def entries() -> list[TimelineEntry]:
    return list(_ENTRIES)


def by_family() -> dict[str, TimelineEntry]:
    return {e.family: e for e in _ENTRIES}
