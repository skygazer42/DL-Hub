"""2D MOT timeline metadata (best-effort, for docs/CLI)."""

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
    TimelineEntry(2016, "sort", "SORT (Kalman + Hungarian baseline)", "online_association"),
    TimelineEntry(2017, "iou_tracker", "IOU Tracker (pure IoU assignment)", "online_association"),
    TimelineEntry(
        2018, "v_iou", "V-IOU (IoU + short visual verification)", "online_association"
    ),
    TimelineEntry(2017, "deepsort", "DeepSORT (SORT + ReID embedding)", "online_association"),
    TimelineEntry(
        2022, "strongsort", "StrongSORT (DeepSORT with stronger association)", "online_association"
    ),
    TimelineEntry(
        2023, "strongsort_pp", "StrongSORT++ (long-occlusion enhanced)", "online_association"
    ),
    TimelineEntry(
        2021,
        "bytetrack",
        "ByteTrack (high/low-score dual-stage association)",
        "online_association",
    ),
    TimelineEntry(2022, "ocsort", "OC-SORT (observation-centric motion model)", "online_association"),
    TimelineEntry(2023, "deep_ocsort", "Deep-OC-SORT (OC-SORT + ReID)", "online_association"),
    TimelineEntry(
        2022,
        "bot_sort",
        "BoT-SORT (ByteTrack + ReID + camera motion compensation)",
        "online_association",
    ),
    TimelineEntry(2018, "motdt", "MOTDT (detection-driven online tracking)", "online_association"),
    TimelineEntry(2024, "fairsort", "FairSORT (balanced online association)", "online_association"),
    TimelineEntry(2022, "rectrack", "RecTrack (recurrent online association)", "online_association"),
    TimelineEntry(2023, "crowdsort", "CrowdSORT (crowd-scene association refinement)", "online_association"),
    TimelineEntry(2024, "hybrid_sort", "Hybrid-SORT (motion + appearance hybrid)", "online_association"),
    TimelineEntry(2021, "uav_sort", "UAV-SORT (motion-prior online tracking)", "online_association"),
    TimelineEntry(2015, "camshift_sort", "CamShift + SORT hybrid", "online_association"),
    TimelineEntry(
        2024, "motionfusion_sort", "MotionFusion-SORT (fused motion association)", "online_association"
    ),
    TimelineEntry(
        2023,
        "velocity_iou_plus",
        "Velocity-IoU+ (velocity-aware IoU assignment)",
        "online_association",
    ),
    TimelineEntry(2018, "tracktor", "Tracktor (regression-based tracking)", "joint_det_embed"),
    TimelineEntry(2021, "tracktor_pp", "Tracktor++ (enhanced Tracktor)", "joint_det_embed"),
    TimelineEntry(
        2020,
        "centertrack",
        "CenterTrack (center-point detection and offset tracking)",
        "joint_det_embed",
    ),
    TimelineEntry(2019, "jde", "JDE (joint detection and embedding)", "joint_det_embed"),
    TimelineEntry(2020, "fairmot", "FairMOT (anchor-free JDE family)", "joint_det_embed"),
    TimelineEntry(2021, "cstrack", "CSTrack (collaborative detection and embedding)", "joint_det_embed"),
    TimelineEntry(
        2021,
        "trades",
        "TraDeS (tracking by temporal-aware detection)",
        "joint_det_embed",
    ),
    TimelineEntry(2021, "qdtrack", "QDTrack (quasi-dense matching)", "joint_det_embed"),
    TimelineEntry(2022, "onetrack", "OneTrack (single-head detection tracking)", "joint_det_embed"),
    TimelineEntry(2021, "siammot", "SiamMOT (Siamese detection-tracking fusion)", "joint_det_embed"),
    TimelineEntry(2021, "fcos_track", "FCOS-Track (anchor-free detection tracking)", "joint_det_embed"),
    TimelineEntry(2022, "yolox_track", "YOLOX-Track (detector-tracker joint head)", "joint_det_embed"),
    TimelineEntry(2023, "d2track", "D2Track (dense-to-dense association)", "joint_det_embed"),
    TimelineEntry(2022, "relationtrack", "RelationTrack (relation-aware embeddings)", "joint_det_embed"),
    TimelineEntry(2020, "reidtrack", "ReIDTrack (joint detector and ReID branch)", "joint_det_embed"),
    TimelineEntry(2019, "masktrack_rcnn", "MaskTrack R-CNN style tracking head", "joint_det_embed"),
    TimelineEntry(2017, "dan_track", "DAN-Track (deep affinity network style)", "joint_det_embed"),
    TimelineEntry(2024, "sparse_reid_track", "Sparse ReID-aware joint tracking", "joint_det_embed"),
    TimelineEntry(2021, "focaltrack", "FocalTrack (focal association objective)", "joint_det_embed"),
    TimelineEntry(
        2021, "transtrack", "TransTrack (DETR style tracking queries)", "query_transformer"
    ),
    TimelineEntry(2022, "trackformer", "TrackFormer (query inheritance over time)", "query_transformer"),
    TimelineEntry(2022, "motr", "MOTR (end-to-end online transformer MOT)", "query_transformer"),
    TimelineEntry(2023, "memotr", "MeMOTR (long-term memory MOTR)", "query_transformer"),
    TimelineEntry(2022, "ctracker", "CTracker (center/query transformer tracking)", "query_transformer"),
    TimelineEntry(2023, "sparsetrack", "SparseTrack (sparse query association)", "query_transformer"),
    TimelineEntry(
        2024,
        "global_transformer_assoc",
        "Global transformer association family",
        "query_transformer",
    ),
    TimelineEntry(2023, "unicorn", "UNICORN (unified video understanding tracking)", "query_transformer"),
    TimelineEntry(2022, "tubetk", "TubeTK (tube-level temporal modeling)", "query_transformer"),
    TimelineEntry(2019, "trackletnet", "TrackletNet (tracklet-level similarity)", "query_transformer"),
    TimelineEntry(2023, "motip", "MOTIP (identity propagation transformer)", "query_transformer"),
    TimelineEntry(2022, "deformtrack", "DeformTrack (deformable-query tracking)", "query_transformer"),
    TimelineEntry(2024, "streamtrack", "StreamTrack (streaming transformer association)", "query_transformer"),
    TimelineEntry(2023, "relationformer_track", "RelationFormer-Track (relation-aware query tracking)", "query_transformer"),
    TimelineEntry(2024, "stq_track", "STQ-Track (spatiotemporal query tracker)", "query_transformer"),
    TimelineEntry(2024, "motrv2", "MOTRv2 (improved transformer query tracker)", "query_transformer"),
    TimelineEntry(2024, "qdetr_track", "QDETR-Track (query-DETR tracking variant)", "query_transformer"),
    TimelineEntry(2023, "track_deformer", "Track-Deformer (deformable trajectory queries)", "query_transformer"),
    TimelineEntry(2023, "tokentrack", "TokenTrack (token-level temporal linking)", "query_transformer"),
    TimelineEntry(2020, "gnn_assoc", "GNN association graph matching", "global_optimization"),
    TimelineEntry(2008, "network_flow", "Network flow data association", "global_optimization"),
    TimelineEntry(2014, "k_shortest_path", "K-shortest path association", "global_optimization"),
    TimelineEntry(2017, "lifted_multicut", "Lifted multicut graph partitioning", "global_optimization"),
    TimelineEntry(
        2018,
        "correlation_clustering",
        "Correlation clustering for global assignment",
        "global_optimization",
    ),
    TimelineEntry(2007, "min_cost_flow", "Min-cost flow data association", "global_optimization"),
    TimelineEntry(2019, "lagrangian_assoc", "Lagrangian-relaxed multi-frame association", "global_optimization"),
    TimelineEntry(2016, "graph_cut_track", "Graph-cut based trajectory partitioning", "global_optimization"),
    TimelineEntry(2020, "mwis_assoc", "MWIS association graph optimization", "global_optimization"),
    TimelineEntry(2018, "benders_flow", "Benders-decomposition flow association", "global_optimization"),
    TimelineEntry(2021, "temporal_clique", "Temporal clique partitioning for trajectories", "global_optimization"),
    TimelineEntry(2022, "graph_stitching", "Graph stitching for fragmented tracklets", "global_optimization"),
    TimelineEntry(1980, "mht", "Multiple Hypothesis Tracking", "probabilistic_filtering"),
    TimelineEntry(1979, "jpda", "Joint Probabilistic Data Association", "probabilistic_filtering"),
    TimelineEntry(2014, "glmb_lmb", "GLMB / LMB random finite set filtering", "probabilistic_filtering"),
    TimelineEntry(2018, "pmbm_gmphd", "PMBM / GM-PHD filtering family", "probabilistic_filtering"),
    TimelineEntry(
        2024,
        "global_hypothesis_bank",
        "Global hypothesis bank for long-horizon MOT",
        "probabilistic_filtering",
    ),
    TimelineEntry(2003, "particle_filter_bank", "Particle-filter track bank", "probabilistic_filtering"),
    TimelineEntry(1999, "rbmht", "Rao-Blackwellized MHT family", "probabilistic_filtering"),
    TimelineEntry(2015, "phd_lmb", "PHD/LMB hybrid random finite set filtering", "probabilistic_filtering"),
    TimelineEntry(2011, "gibbs_jpda", "Gibbs-sampled JPDA approximation", "probabilistic_filtering"),
    TimelineEntry(
        2017,
        "bernoulli_mixture_track",
        "Bernoulli-mixture multi-object filtering",
        "probabilistic_filtering",
    ),
    TimelineEntry(2020, "variational_mht", "Variational inference for MHT", "probabilistic_filtering"),
]


def entries() -> list[TimelineEntry]:
    return list(_ENTRIES)


def by_family() -> dict[str, TimelineEntry]:
    return {e.family: e for e in _ENTRIES}
