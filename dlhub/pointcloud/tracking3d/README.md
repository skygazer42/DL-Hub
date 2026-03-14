# PointCloud Tracking3D Local Zoo

This directory provides **130 local 3D tracking families** with `tiny/small/base` variants
(390 architecture IDs in total).

- Arch ID format: `pctrk3d:<family>_<variant>`
- Unified builder: `dlhub.pointcloud.tracking3d_zoo.build_local_model(...)`
- CLI entry: `python scripts/tracking3d_zoo.py`

## Quick Commands

```bash
python scripts/tracking3d_zoo.py --list
python scripts/tracking3d_zoo.py --timeline
python scripts/tracking3d_zoo.py --smoke pctrk3d:ab3dmot_tiny

python scripts/tracking3d_zoo.py --list-profiles
python scripts/tracking3d_zoo.py --recommend bev_priority --top-k 8 --variant tiny
python scripts/tracking3d_zoo.py --recommend segmentation_first --top-k 8 --variant tiny --emit-smoke-cmds
python scripts/tracking3d_zoo.py --recommend bev_priority --top-k 3 --variant tiny --run-smoke-cmds --summary-only
python scripts/tracking3d_zoo.py --recommend long_horizon --top-k 3 --variant tiny --run-smoke-cmds --rank-by elapsed
python scripts/tracking3d_zoo.py --recommend bev_priority --top-k 3 --variant tiny --run-smoke-cmds --save-artifacts-dir outputs/pointcloud/tracking3d_artifacts
python scripts/tracking3d_zoo.py --recommend bev_priority --top-k 3 --variant tiny --run-smoke-cmds --save-artifacts-dir auto
```

## 130 Families by Group

| Group | Families |
|---|---|
| `kalman_association` | `ab3dmot`, `simpletrack`, `imm_kalman`, `ocsort3d`, `deepsort3d`, `ma3dmot`, `ukf3d`, `ekf3d`, `lidar_iou_track`, `gnn_kalman3d`, `strongsort3d`, `tracklet_kf3d`, `adaptive_kf3d`, `mahalanobis3d`, `probabilistic_iou3d` |
| `bev_tracking` | `centerpoint_track`, `bitrack`, `bevsort`, `bevfusion_track`, `voxeltrack`, `centertrack3d`, `pillartrack`, `transcenter3d`, `centerbev_track`, `motionbev_track`, `querybev_track`, `sparsebev_track`, `mapbev_track`, `hdmap_bev_track`, `lanebev_track`, `occupancy_bev_track`, `temporalbev_track`, `velocitybev_track`, `scenebev_track`, `multimodal_bev_track`, `anchorfree_bev_track`, `transformbev_track`, `streambev_track`, `bevformer_track`, `bevnext_track`, `depthbev_track`, `graphbev_track`, `memorybev_track`, `radarbev_track`, `stereo_bev_track`, `trajectorybev_track`, `uncertaintybev_track`, `worldbev_track`, `mapprior_bev_track`, `vectorbev_track`, `crossview_bev_track`, `liftbev_track`, `occupancyflow_bev_track`, `sparseformer_bev_track`, `eventbev_track`, `planningbev_track`, `topologybev_track`, `geobev_track`, `cambev_track`, `lidarbev_track`, `radarfusion_bev_track`, `maplane_bev_track`, `scenegraph_bev_track`, `interactivebev_track`, `predictivebev_track`, `globalbev_track`, `hyperbev_track`, `robustbev_track`, `lowlatency_bev_track`, `tinybev_track`, `quantbev_track`, `edgebev_track`, `compressedbev_track`, `distillbev_track`, `mobilebev_track`, `fastmap_bev_track`, `agilebev_track`, `streamlite_bev_track`, `ultrafast_bev_track`, `realtime_bev_track`, `nanobev_track`, `microbev_track`, `econobev_track`, `slimbev_track`, `swiftbev_track`, `powerbev_track`, `budgetbev_track`, `turbo_bev_track`, `sensorlite_bev_track`, `ondevice_bev_track`, `lowpower_bev_track`, `cachebev_track`, `instantbev_track`, `rapidbev_track`, `frugalbev_track`, `compactbev_track`, `sparselite_bev_track`, `latencyguard_bev_track`, `ultralite_bev_track`, `minipower_bev_track`, `featherbev_track`, `scoutbev_track`, `zipbev_track`, `thriftbev_track`, `flashbev_track`, `zipstream_bev_track`, `quickmap_bev_track`, `nanoedge_bev_track`, `pulsebev_track`, `briskbev_track`, `sprintbev_track`, `leanbev_track`, `rangerbev_track`, `depotbev_track`, `meshbev_track`, `relaybev_track`, `nimblebev_track`, `steadyedge_bev_track` |
| `segmentation_tracking` | `motsf3d`, `pointtrack3d`, `masktrack3d`, `segtrack3d`, `panoptictrack3d`, `instanceflow3d`, `trackletseg3d`, `maskprop3d`, `voxelmask_track3d`, `semtrack3d`, `objectflow3d`, `dynseg_track3d` |

## Recommendation Profiles

| Profile | Focus |
|---|---|
| `balanced` | Mix Kalman + BEV + segmentation families |
| `realtime_lidar` | Lightweight LiDAR online tracking |
| `bev_priority` | BEV-first tracking for vehicle pipelines |
| `segmentation_first` | Segmentation-guided 3D tracking |
| `long_horizon` | Long-horizon stability and continuity |
