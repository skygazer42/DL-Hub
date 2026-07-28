# Vision MOT Local Zoo (2D, Single-Camera)

本目录提供 **80 个 MOT 算法族** 的 compact-first 本地实现（纯 PyTorch、无需下载权重），统一通过 `mot2d:<family>_<variant>` 调用：
- 变体：`tiny / small / base`
- 统一入口：`dlhub.vision.mot_zoo.build_local_model(...)`
- CLI：`python scripts/mot_zoo.py --list|--timeline|--smoke`

## 快速命令
```bash
python scripts/mot_zoo.py --list
python scripts/mot_zoo.py --search bytetrack
python scripts/mot_zoo.py --timeline
python scripts/mot_zoo.py --list-profiles
python scripts/mot_zoo.py --recommend realtime --top-k 8 --variant tiny
python scripts/mot_zoo.py --recommend occlusion --top-k 8 --variant tiny --emit-train-cmds
python scripts/mot_zoo.py --recommend realtime --top-k 3 --variant tiny --run-train-cmds
python scripts/mot_zoo.py --recommend realtime --top-k 3 --variant tiny --run-train-cmds --skip-existing
python scripts/mot_zoo.py --recommend realtime --top-k 3 --variant tiny --run-train-cmds --summary-only
python scripts/mot_zoo.py --recommend realtime --top-k 3 --variant tiny --run-train-cmds --rank-by acc
python scripts/mot_zoo.py --recommend realtime --top-k 3 --variant tiny --run-train-cmds --save-leaderboard outputs/vision/mot_leaderboard.csv
python scripts/mot_zoo.py --recommend realtime --top-k 3 --variant tiny --run-train-cmds --save-artifacts-dir outputs/vision/mot_artifacts
python scripts/mot_zoo.py --recommend realtime --top-k 3 --variant tiny --run-train-cmds --save-artifacts-dir auto
python scripts/mot_zoo.py --smoke mot2d:sort_tiny
```

## 80 算法族分组
| 组别 | 算法族 |
|---|---|
| `online_association` | `sort`, `iou_tracker`, `v_iou`, `deepsort`, `strongsort`, `strongsort_pp`, `bytetrack`, `ocsort`, `deep_ocsort`, `bot_sort`, `motdt`, `fairsort`, `rectrack`, `crowdsort`, `hybrid_sort`, `uav_sort`, `camshift_sort`, `motionfusion_sort`, `velocity_iou_plus` |
| `joint_det_embed` | `tracktor`, `tracktor_pp`, `centertrack`, `jde`, `fairmot`, `cstrack`, `trades`, `qdtrack`, `onetrack`, `siammot`, `fcos_track`, `yolox_track`, `d2track`, `relationtrack`, `reidtrack`, `masktrack_rcnn`, `dan_track`, `sparse_reid_track`, `focaltrack` |
| `query_transformer` | `transtrack`, `trackformer`, `motr`, `memotr`, `ctracker`, `sparsetrack`, `global_transformer_assoc`, `unicorn`, `tubetk`, `trackletnet`, `motip`, `deformtrack`, `streamtrack`, `relationformer_track`, `stq_track`, `motrv2`, `qdetr_track`, `track_deformer`, `tokentrack` |
| `global_optimization` | `gnn_assoc`, `network_flow`, `k_shortest_path`, `lifted_multicut`, `correlation_clustering`, `min_cost_flow`, `lagrangian_assoc`, `graph_cut_track`, `mwis_assoc`, `benders_flow`, `temporal_clique`, `graph_stitching` |
| `probabilistic_filtering` | `mht`, `jpda`, `glmb_lmb`, `pmbm_gmphd`, `global_hypothesis_bank`, `particle_filter_bank`, `rbmht`, `phd_lmb`, `gibbs_jpda`, `bernoulli_mixture_track`, `variational_mht` |

## 选型建议（先跑通再细化）
- 在线关联优先：`mot2d:sort_tiny`、`mot2d:bytetrack_tiny`、`mot2d:motdt_tiny`、`mot2d:uav_sort_tiny`
- 检测 + ReID 联合：`mot2d:jde_tiny`、`mot2d:fairmot_tiny`、`mot2d:reidtrack_tiny`、`mot2d:masktrack_rcnn_tiny`
- Transformer 查询关联：`mot2d:transtrack_tiny`、`mot2d:motr_tiny`、`mot2d:motip_tiny`、`mot2d:motrv2_tiny`
- 全局优化方向：`mot2d:gnn_assoc_tiny`、`mot2d:network_flow_tiny`、`mot2d:min_cost_flow_tiny`、`mot2d:mwis_assoc_tiny`
- 概率滤波方向：`mot2d:mht_tiny`、`mot2d:jpda_tiny`、`mot2d:rbmht_tiny`、`mot2d:variational_mht_tiny`

> 时间线元数据见 `dlhub/vision/mot/_timeline.py`。
