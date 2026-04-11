# PointCloud Scene Flow Local Zoo

This directory provides **10 local scene-flow families** with `tiny/small/base` variants
(30 architecture IDs in total).

- Arch ID format: `pcsf3d:<family>_<variant>`
- Unified builder: `dlhub.pointcloud.scene_flow_zoo.build_local_model(...)`
- Lazy package: `dlhub.pointcloud.scene_flow`

## Included Families

- `flow3d_pointnet`
- `flow3d_flownet`
- `pointpwc_flow`
- `raft3d_points`
- `iter_flow3d`
- `cost_volume_flow3d`
- `transformer_flow3d`
- `diffusion_flow3d`
- `prompt_flow3d`
- `mamba_flow3d`

## Quick Commands

```bash
python -c "from dlhub.pointcloud.scene_flow_zoo import list_local_arches; print(list_local_arches())"
python -c "from dlhub.pointcloud.scene_flow_zoo import build_local_model; print(type(build_local_model('pcsf3d:flow3d_pointnet_tiny', in_channels=3)).__name__)"
python -m dlhub.pointcloud.scene_flow.flow3d_pointnet
```
