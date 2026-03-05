# Pointcloud 3D Detection Zoo (40 算法族) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 为 DL-Hub 增加一个「纯 torch、toy-first、CPU 友好」的 **3D 目标检测 (point cloud / BEV)** 本地模型 zoo，并提供可枚举/可构建的 arch id、pytest 冒烟测试和脚本工具。

**Architecture:** 新增 `dlhub/pointcloud/detection3d/` 作为算法族目录（**一算法族一文件**，变种写在 `_VARIANTS` 里），`dlhub/pointcloud/detection3d_zoo.py` 负责从源码提取变种并做 lazy builder，避免导入开销；`tests/` 提供 tiny 变种的 forward+backward 冒烟验证。

**Tech Stack:** Python + PyTorch（仅 `torch`/`torch.nn`，不引入外部模型包；实现以可读/可跑为优先）。

---

## I/O 约定（Toy）

- **Input**: `points` tensor, shape `(B, N, C)`，其中 `C>=3`，前 3 维为 `xyz`。
- **Output**: `dict` 至少包含：
  - `boxes`: `(B, K, 7)` -> `(x, y, z, dx, dy, dz, yaw)`
  - `cls_logits`: `(B, K, num_classes)`

---

## Files & Layout

### Core package
- Create: `dlhub/pointcloud/detection3d/__init__.py`
- Create: `dlhub/pointcloud/detection3d/_common.py`
  - toy blocks: PointNet/EdgeConv、BEV scatter、BEV dense head、top-k 解码、ROI kNN pooling、tiny transformer 等

### Zoo / CLI tooling
- Create: `dlhub/pointcloud/detection3d_zoo.py`
  - 从 `dlhub/pointcloud/detection3d/*.py` 源码提取 `_VARIANTS` + `build_*_detector3d`
  - arch id prefix: `pcdet3d:*`
- Create: `scripts/detection3d_zoo.py`
  - `--list` / `--search` / `--smoke`（随机点云 forward）

### Tests
- Create: `tests/test_dlhub_pointcloud_detection3d_zoo.py`
  - 确保 arch 数量 >= 120（40 算法族 × 3 tiny/small/base）
  - 对所有 `*_tiny` arch 做 forward + backward smoke

---

## 40 算法族（文件清单）

BEV / voxel / pillar:
- `voxelnet.py`, `second.py`, `pointpillars.py`, `pillarnet.py`, `centerpoint.py`
- `pixor.py`, `complexyolo.py`, `hotspotnet.py`, `afdet.py`, `voxelnext.py`
- `votr.py`, `sst.py`, `tanet.py`, `bevfusion.py`, `transfusion.py`

Two-stage / proposal + ROI:
- `pv_rcnn.py`, `pv_rcnn_pp.py`, `voxel_rcnn.py`, `point_rcnn.py`, `parta2.py`
- `avod.py`, `mv3d.py`, `frustum_pointnet.py`

Indoor / vote / transformer:
- `votenet.py`, `h3dnet.py`, `imvotenet.py`, `groupfree3d.py`, `threedetr.py`

Point / graph / range-view:
- `pointgnn.py`, `lasernet.py`

Single-stage keypoints:
- `threedssd.py`, `sassd.py`, `ciassd.py`, `iassd.py`, `sessd.py`

3D conv dense:
- `fcaf3d.py`

Baselines:
- `pointnet_det.py`, `dgcnn_det.py`

---

## Verification Commands

- Run unit tests: `pytest -q`
- List arch ids: `python scripts/detection3d_zoo.py --list`
- Smoke one model: `python scripts/detection3d_zoo.py --smoke pcdet3d:pointpillars_tiny`

