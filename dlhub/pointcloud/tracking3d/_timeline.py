"""3D tracking timeline metadata (best-effort, for docs/CLI)."""

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
    TimelineEntry(2020, "ab3dmot", "AB3DMOT (Kalman + 3D box association)", "kalman_association"),
    TimelineEntry(
        2021, "centerpoint_track", "CenterPoint-Track (BEV center tracking)", "bev_tracking"
    ),
    TimelineEntry(
        2022, "simpletrack", "SimpleTrack (lightweight Kalman + affinity)", "kalman_association"
    ),
    TimelineEntry(2023, "bitrack", "BiTrack (bi-directional BEV association)", "bev_tracking"),
    TimelineEntry(
        2020, "motsf3d", "MOTSF3D (joint 3D segmentation/tracking)", "segmentation_tracking"
    ),
    TimelineEntry(
        2019, "imm_kalman", "IMM-Kalman (multi-model motion tracking)", "kalman_association"
    ),
    TimelineEntry(
        2022, "ocsort3d", "OCSORT3D (observation-centric 3D association)", "kalman_association"
    ),
    TimelineEntry(2021, "deepsort3d", "DeepSORT3D (appearance-aware 3D tracking)", "kalman_association"),
    TimelineEntry(2023, "ma3dmot", "MA3DMOT (motion-appearance 3D association)", "kalman_association"),
    TimelineEntry(2018, "ukf3d", "UKF3D (unscented Kalman tracking)", "kalman_association"),
    TimelineEntry(2017, "ekf3d", "EKF3D (extended Kalman tracking)", "kalman_association"),
    TimelineEntry(
        2020, "lidar_iou_track", "LiDAR IoU Track (3D IoU association)", "kalman_association"
    ),
    TimelineEntry(2021, "bevsort", "BEVSORT (BEV association baseline)", "bev_tracking"),
    TimelineEntry(2023, "bevfusion_track", "BEVFusion-Track (fusion-aware BEV tracking)", "bev_tracking"),
    TimelineEntry(2022, "voxeltrack", "VoxelTrack (voxel-based BEV tracking)", "bev_tracking"),
    TimelineEntry(2023, "centertrack3d", "CenterTrack3D (3D center heatmap tracking)", "bev_tracking"),
    TimelineEntry(2022, "pillartrack", "PillarTrack (point-pillar BEV tracking)", "bev_tracking"),
    TimelineEntry(2021, "transcenter3d", "TransCenter3D (transformer BEV center tracking)", "bev_tracking"),
    TimelineEntry(2022, "pointtrack3d", "PointTrack3D (point-level instance association)", "segmentation_tracking"),
    TimelineEntry(2021, "masktrack3d", "MaskTrack3D (mask-guided 3D tracking)", "segmentation_tracking"),
    TimelineEntry(2020, "segtrack3d", "SegTrack3D (segmentation-coupled tracking)", "segmentation_tracking"),
    TimelineEntry(2023, "panoptictrack3d", "PanopticTrack3D (panoptic 3D tracking)", "segmentation_tracking"),
    TimelineEntry(
        2022,
        "instanceflow3d",
        "InstanceFlow3D (instance flow guided tracking)",
        "segmentation_tracking",
    ),
    TimelineEntry(
        2021, "trackletseg3d", "TrackletSeg3D (tracklet segmentation linking)", "segmentation_tracking"
    ),
    TimelineEntry(
        2024, "gnn_kalman3d", "GNN-Kalman3D (graph-assisted Kalman association)", "kalman_association"
    ),
    TimelineEntry(
        2023, "strongsort3d", "StrongSORT3D (strong appearance Kalman tracker)", "kalman_association"
    ),
    TimelineEntry(
        2024, "tracklet_kf3d", "Tracklet-KF3D (tracklet-level Kalman fusion)", "kalman_association"
    ),
    TimelineEntry(
        2025, "adaptive_kf3d", "Adaptive-KF3D (adaptive Kalman motion model)", "kalman_association"
    ),
    TimelineEntry(
        2024, "mahalanobis3d", "Mahalanobis3D (distance-aware Kalman gating)", "kalman_association"
    ),
    TimelineEntry(
        2025,
        "probabilistic_iou3d",
        "Probabilistic-IoU3D (uncertainty-aware IoU association)",
        "kalman_association",
    ),
    TimelineEntry(2024, "centerbev_track", "CenterBEV-Track (center-guided BEV tracking)", "bev_tracking"),
    TimelineEntry(2025, "motionbev_track", "MotionBEV-Track (motion-prior BEV tracking)", "bev_tracking"),
    TimelineEntry(2025, "querybev_track", "QueryBEV-Track (query-based BEV tracking)", "bev_tracking"),
    TimelineEntry(2024, "sparsebev_track", "SparseBEV-Track (sparse token BEV tracking)", "bev_tracking"),
    TimelineEntry(2025, "mapbev_track", "MapBEV-Track (map-aware BEV tracking)", "bev_tracking"),
    TimelineEntry(
        2025, "hdmap_bev_track", "HDMap-BEV-Track (HD-map conditioned BEV tracking)", "bev_tracking"
    ),
    TimelineEntry(2025, "lanebev_track", "LaneBEV-Track (lane-aware BEV tracking)", "bev_tracking"),
    TimelineEntry(
        2026, "occupancy_bev_track", "Occupancy-BEV-Track (occupancy-guided tracking)", "bev_tracking"
    ),
    TimelineEntry(
        2026, "temporalbev_track", "TemporalBEV-Track (temporal memory BEV tracking)", "bev_tracking"
    ),
    TimelineEntry(
        2026, "velocitybev_track", "VelocityBEV-Track (velocity-aware BEV tracking)", "bev_tracking"
    ),
    TimelineEntry(
        2026, "scenebev_track", "SceneBEV-Track (scene-context BEV tracking)", "bev_tracking"
    ),
    TimelineEntry(
        2026,
        "multimodal_bev_track",
        "Multimodal-BEV-Track (multi-sensor BEV fusion tracking)",
        "bev_tracking",
    ),
    TimelineEntry(
        2026,
        "anchorfree_bev_track",
        "AnchorFree-BEV-Track (anchor-free BEV association)",
        "bev_tracking",
    ),
    TimelineEntry(
        2026,
        "transformbev_track",
        "TransformBEV-Track (transformer-enhanced BEV tracking)",
        "bev_tracking",
    ),
    TimelineEntry(2026, "streambev_track", "StreamBEV-Track (online stream BEV tracking)", "bev_tracking"),
    TimelineEntry(2026, "bevformer_track", "BEVFormer-Track (spatiotemporal BEV transformer)", "bev_tracking"),
    TimelineEntry(2026, "bevnext_track", "BEVNeXt-Track (next-gen BEV association)", "bev_tracking"),
    TimelineEntry(
        2026, "depthbev_track", "DepthBEV-Track (depth-aware BEV tracking)", "bev_tracking"
    ),
    TimelineEntry(2026, "graphbev_track", "GraphBEV-Track (graph relational BEV tracking)", "bev_tracking"),
    TimelineEntry(2026, "memorybev_track", "MemoryBEV-Track (memory-augmented BEV tracking)", "bev_tracking"),
    TimelineEntry(2026, "radarbev_track", "RadarBEV-Track (radar-lidar BEV fusion tracking)", "bev_tracking"),
    TimelineEntry(2026, "stereo_bev_track", "Stereo-BEV-Track (stereo BEV projection tracking)", "bev_tracking"),
    TimelineEntry(
        2026,
        "trajectorybev_track",
        "TrajectoryBEV-Track (trajectory-prior BEV tracking)",
        "bev_tracking",
    ),
    TimelineEntry(
        2026,
        "uncertaintybev_track",
        "UncertaintyBEV-Track (uncertainty-calibrated BEV tracking)",
        "bev_tracking",
    ),
    TimelineEntry(2026, "worldbev_track", "WorldBEV-Track (world-model BEV tracking)", "bev_tracking"),
    TimelineEntry(2026, "mapprior_bev_track", "MapPrior-BEV-Track (map-prior BEV tracking)", "bev_tracking"),
    TimelineEntry(2026, "vectorbev_track", "VectorBEV-Track (vectorized scene BEV tracking)", "bev_tracking"),
    TimelineEntry(
        2026,
        "crossview_bev_track",
        "CrossView-BEV-Track (cross-view fused BEV tracking)",
        "bev_tracking",
    ),
    TimelineEntry(2026, "liftbev_track", "LiftBEV-Track (lift-splat BEV tracking)", "bev_tracking"),
    TimelineEntry(
        2026,
        "occupancyflow_bev_track",
        "OccupancyFlow-BEV-Track (occupancy flow BEV tracking)",
        "bev_tracking",
    ),
    TimelineEntry(
        2026,
        "sparseformer_bev_track",
        "SparseFormer-BEV-Track (sparse transformer BEV tracking)",
        "bev_tracking",
    ),
    TimelineEntry(2026, "eventbev_track", "EventBEV-Track (event-guided BEV tracking)", "bev_tracking"),
    TimelineEntry(2026, "planningbev_track", "PlanningBEV-Track (planning-aware BEV tracking)", "bev_tracking"),
    TimelineEntry(2026, "topologybev_track", "TopologyBEV-Track (topology-aware BEV tracking)", "bev_tracking"),
    TimelineEntry(2026, "geobev_track", "GeoBEV-Track (geometric prior BEV tracking)", "bev_tracking"),
    TimelineEntry(2026, "cambev_track", "CamBEV-Track (camera-centric BEV tracking)", "bev_tracking"),
    TimelineEntry(2026, "lidarbev_track", "LiDARBEV-Track (lidar-centric BEV tracking)", "bev_tracking"),
    TimelineEntry(
        2026,
        "radarfusion_bev_track",
        "RadarFusion-BEV-Track (radar-camera-lidar BEV fusion)",
        "bev_tracking",
    ),
    TimelineEntry(2026, "maplane_bev_track", "MapLane-BEV-Track (lane-map aware BEV tracking)", "bev_tracking"),
    TimelineEntry(
        2026,
        "scenegraph_bev_track",
        "SceneGraph-BEV-Track (scene graph guided BEV tracking)",
        "bev_tracking",
    ),
    TimelineEntry(
        2026,
        "interactivebev_track",
        "InteractiveBEV-Track (interactive agent BEV tracking)",
        "bev_tracking",
    ),
    TimelineEntry(
        2026,
        "predictivebev_track",
        "PredictiveBEV-Track (prediction coupled BEV tracking)",
        "bev_tracking",
    ),
    TimelineEntry(
        2026,
        "globalbev_track",
        "GlobalBEV-Track (global context BEV tracking)",
        "bev_tracking",
    ),
    TimelineEntry(2026, "hyperbev_track", "HyperBEV-Track (hypernetwork BEV tracking)", "bev_tracking"),
    TimelineEntry(2026, "robustbev_track", "RobustBEV-Track (robustness focused BEV tracking)", "bev_tracking"),
    TimelineEntry(
        2026, "lowlatency_bev_track", "LowLatency-BEV-Track (low-latency BEV tracking)", "bev_tracking"
    ),
    TimelineEntry(2026, "tinybev_track", "TinyBEV-Track (tiny footprint BEV tracking)", "bev_tracking"),
    TimelineEntry(2026, "quantbev_track", "QuantBEV-Track (quantized BEV tracking)", "bev_tracking"),
    TimelineEntry(2026, "edgebev_track", "EdgeBEV-Track (edge deployment BEV tracking)", "bev_tracking"),
    TimelineEntry(
        2026,
        "compressedbev_track",
        "CompressedBEV-Track (compressed representation BEV tracking)",
        "bev_tracking",
    ),
    TimelineEntry(2026, "distillbev_track", "DistillBEV-Track (distilled BEV tracking)", "bev_tracking"),
    TimelineEntry(2026, "mobilebev_track", "MobileBEV-Track (mobile optimized BEV tracking)", "bev_tracking"),
    TimelineEntry(
        2026, "fastmap_bev_track", "FastMap-BEV-Track (fast map-aware BEV tracking)", "bev_tracking"
    ),
    TimelineEntry(2026, "agilebev_track", "AgileBEV-Track (agile update BEV tracking)", "bev_tracking"),
    TimelineEntry(
        2026,
        "streamlite_bev_track",
        "StreamLite-BEV-Track (lightweight stream BEV tracking)",
        "bev_tracking",
    ),
    TimelineEntry(
        2026,
        "ultrafast_bev_track",
        "UltraFast-BEV-Track (ultra-fast BEV inference tracking)",
        "bev_tracking",
    ),
    TimelineEntry(
        2026,
        "realtime_bev_track",
        "RealTime-BEV-Track (real-time constrained BEV tracking)",
        "bev_tracking",
    ),
    TimelineEntry(2026, "nanobev_track", "NanoBEV-Track (nano scale BEV tracker)", "bev_tracking"),
    TimelineEntry(2026, "microbev_track", "MicroBEV-Track (micro footprint BEV tracker)", "bev_tracking"),
    TimelineEntry(2026, "econobev_track", "EconoBEV-Track (cost-efficient BEV tracking)", "bev_tracking"),
    TimelineEntry(2026, "slimbev_track", "SlimBEV-Track (slim model BEV tracking)", "bev_tracking"),
    TimelineEntry(2026, "swiftbev_track", "SwiftBEV-Track (swift response BEV tracking)", "bev_tracking"),
    TimelineEntry(2026, "powerbev_track", "PowerBEV-Track (compute-rich BEV tracking)", "bev_tracking"),
    TimelineEntry(2026, "budgetbev_track", "BudgetBEV-Track (budget-friendly BEV tracking)", "bev_tracking"),
    TimelineEntry(2026, "turbo_bev_track", "Turbo-BEV-Track (turbo BEV throughput tracking)", "bev_tracking"),
    TimelineEntry(
        2026,
        "sensorlite_bev_track",
        "SensorLite-BEV-Track (sensor-lightweight BEV tracking)",
        "bev_tracking",
    ),
    TimelineEntry(
        2026, "ondevice_bev_track", "OnDevice-BEV-Track (on-device BEV tracking)", "bev_tracking"
    ),
    TimelineEntry(
        2026, "lowpower_bev_track", "LowPower-BEV-Track (low-power BEV tracking)", "bev_tracking"
    ),
    TimelineEntry(
        2026, "cachebev_track", "CacheBEV-Track (cache-aware BEV tracking)", "bev_tracking"
    ),
    TimelineEntry(
        2026, "instantbev_track", "InstantBEV-Track (instant response BEV tracking)", "bev_tracking"
    ),
    TimelineEntry(2026, "rapidbev_track", "RapidBEV-Track (rapid cycle BEV tracking)", "bev_tracking"),
    TimelineEntry(
        2026, "frugalbev_track", "FrugalBEV-Track (frugal compute BEV tracking)", "bev_tracking"
    ),
    TimelineEntry(
        2026, "compactbev_track", "CompactBEV-Track (compact architecture BEV tracking)", "bev_tracking"
    ),
    TimelineEntry(
        2026, "sparselite_bev_track", "SparseLite-BEV-Track (sparse-lite BEV tracking)", "bev_tracking"
    ),
    TimelineEntry(
        2026,
        "latencyguard_bev_track",
        "LatencyGuard-BEV-Track (latency-guarded BEV tracking)",
        "bev_tracking",
    ),
    TimelineEntry(
        2026, "ultralite_bev_track", "UltraLite-BEV-Track (ultra-light BEV tracking)", "bev_tracking"
    ),
    TimelineEntry(
        2026,
        "minipower_bev_track",
        "MiniPower-BEV-Track (mini-power BEV tracking)",
        "bev_tracking",
    ),
    TimelineEntry(
        2026, "featherbev_track", "FeatherBEV-Track (featherweight BEV tracking)", "bev_tracking"
    ),
    TimelineEntry(
        2026, "scoutbev_track", "ScoutBEV-Track (scout-class BEV tracking)", "bev_tracking"
    ),
    TimelineEntry(2026, "zipbev_track", "ZipBEV-Track (zippy BEV tracking)", "bev_tracking"),
    TimelineEntry(
        2026, "thriftbev_track", "ThriftBEV-Track (thrifty BEV tracking)", "bev_tracking"
    ),
    TimelineEntry(2026, "flashbev_track", "FlashBEV-Track (flash-speed BEV tracking)", "bev_tracking"),
    TimelineEntry(
        2026,
        "zipstream_bev_track",
        "ZipStream-BEV-Track (zip-stream BEV tracking)",
        "bev_tracking",
    ),
    TimelineEntry(
        2026, "quickmap_bev_track", "QuickMap-BEV-Track (quick map BEV tracking)", "bev_tracking"
    ),
    TimelineEntry(
        2026, "nanoedge_bev_track", "NanoEdge-BEV-Track (nano edge BEV tracking)", "bev_tracking"
    ),
    TimelineEntry(2026, "pulsebev_track", "PulseBEV-Track (pulse-optimized BEV tracking)", "bev_tracking"),
    TimelineEntry(2026, "briskbev_track", "BriskBEV-Track (brisk response BEV tracking)", "bev_tracking"),
    TimelineEntry(2026, "sprintbev_track", "SprintBEV-Track (sprint-latency BEV tracking)", "bev_tracking"),
    TimelineEntry(2026, "leanbev_track", "LeanBEV-Track (lean compute BEV tracking)", "bev_tracking"),
    TimelineEntry(2026, "rangerbev_track", "RangerBEV-Track (range-aware BEV tracking)", "bev_tracking"),
    TimelineEntry(2026, "depotbev_track", "DepotBEV-Track (depot-grade BEV tracking)", "bev_tracking"),
    TimelineEntry(2026, "meshbev_track", "MeshBEV-Track (mesh relation BEV tracking)", "bev_tracking"),
    TimelineEntry(2026, "relaybev_track", "RelayBEV-Track (relay stream BEV tracking)", "bev_tracking"),
    TimelineEntry(2026, "nimblebev_track", "NimbleBEV-Track (nimble update BEV tracking)", "bev_tracking"),
    TimelineEntry(
        2026, "steadyedge_bev_track", "SteadyEdge-BEV-Track (steady edge BEV tracking)", "bev_tracking"
    ),
    TimelineEntry(2024, "maskprop3d", "MaskProp3D (mask propagation for 3D tracking)", "segmentation_tracking"),
    TimelineEntry(
        2025,
        "voxelmask_track3d",
        "VoxelMask-Track3D (voxel mask-driven tracking)",
        "segmentation_tracking",
    ),
    TimelineEntry(
        2024, "semtrack3d", "SemTrack3D (semantic segmentation tracking)", "segmentation_tracking"
    ),
    TimelineEntry(
        2025, "objectflow3d", "ObjectFlow3D (instance flow plus segmentation)", "segmentation_tracking"
    ),
    TimelineEntry(
        2025, "dynseg_track3d", "DynSeg-Track3D (dynamic scene segmentation tracking)", "segmentation_tracking"
    ),
]


def entries() -> list[TimelineEntry]:
    return list(_ENTRIES)


def by_family() -> dict[str, TimelineEntry]:
    return {e.family: e for e in _ENTRIES}
