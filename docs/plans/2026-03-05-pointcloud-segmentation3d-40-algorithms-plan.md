# Pointcloud 3D Semantic Segmentation Zoo (40 算法族) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 为 DL-Hub 增加一个「纯 torch、toy-first、CPU 友好」的 **3D 点云语义分割** 本地模型 zoo：支持枚举/构建 arch id、提供 CLI 脚本和 pytest forward+backward 冒烟测试。

**Architecture:** 新增 `dlhub/pointcloud/segmentation3d/` 作为算法族目录（**一算法族一文件**，变种写在 `_VARIANTS` 里），`dlhub/pointcloud/segmentation3d_zoo.py` 用 AST 从源码提取变种并 lazy 构建，避免 import 负担；所有模型统一 I/O（`(B,N,C)` -> `(B,N,num_classes)`）。

**Tech Stack:** Python + PyTorch（仅 `torch`/`torch.nn`；不引入外部点云/稀疏卷积库；实现以可读/可跑为优先）。

---

## I/O 约定（Toy）

- **Input**: `points` tensor, shape `(B, N, C)`，其中 `C>=3`，前 3 维为 `xyz`，其余可选为点特征。
- **Output**: `logits` tensor, shape `(B, N, num_classes)`（每点语义分类 logits）。

---

## Files & Layout

### Core package
- Create: `dlhub/pointcloud/segmentation3d/__init__.py`（lazy builder import）
- Create: `dlhub/pointcloud/segmentation3d/_common.py`
  - toy blocks: Point MLP、EdgeConv、PointNet++ SA/FP、tiny transformer
  - 投影/体素辅助: 2D/3D scatter+gather、TinyUNet2D/TinyUNet3D、Point-Voxel fusion

### Zoo / CLI tooling
- Create: `dlhub/pointcloud/segmentation3d_zoo.py`
  - 从 `dlhub/pointcloud/segmentation3d/*.py` 源码提取 `_VARIANTS` + `build_*_segmenter3d`
  - arch id prefix: `pcseg3d:*`
- Create: `scripts/segmentation3d_zoo.py`
  - `--list / --search / --smoke`（随机点云 forward）

### Tests
- Create: `tests/test_dlhub_pointcloud_segmentation3d_zoo.py`
  - 确保 arch 数量 >= 120（40 算法族 × 3 tiny/small/base）
  - 对所有 `*_tiny` arch 做 forward + backward smoke

---

## 40 算法族（文件清单）

Point-based / graph / transformer:
- `pointnet.py`, `pointnet2.py`, `dgcnn.py`, `kpconv.py`, `pointconv.py`
- `pointcnn.py`, `pvcnn.py`, `pointweb.py`, `pointgcn.py`, `pointgat.py`
- `paconv.py`, `rscnn.py`, `spidercnn.py`, `shellnet.py`, `pointsift.py`
- `curvenet.py`, `gdanet.py`, `asnl.py`, `pointmlp.py`, `pointnext.py`
- `point_transformer.py`, `pct.py`, `pointmixer.py`, `simpleview.py`
- `pointbert.py`, `pointmae.py`, `point2seq.py`, `stratified_transformer.py`, `pointformer.py`

Projection / voxel:
- `cylinder3d.py`, `polarnet.py`, `rangenetpp.py`, `salsanext.py`, `squeezeseg.py`
- `bevunet.py`, `rangeformer.py`, `minkunet.py`, `spvcnn.py`, `voxelunet.py`

---

## Verification Commands

- Run unit tests: `pytest -q`
- List arch ids: `python scripts/segmentation3d_zoo.py --list`
- Smoke one model: `python scripts/segmentation3d_zoo.py --smoke pcseg3d:pointnet_tiny`

