# Pointcloud 3D Instance Segmentation Zoo (30 算法族) Plan

**Goal:** 为 DL-Hub 增加一个「纯 torch、toy-first、CPU 友好」的 **3D 实例分割 (point cloud)** 本地模型 zoo：可枚举/可构建的 arch id、pytest 冒烟测试与脚本工具。

**Architecture:**

- 算法族目录：`dlhub/pointcloud/instance_segmentation3d/`
  - **一算法族一文件**（变种写在 `_VARIANTS` 里：tiny/small/base）
  - 每个文件暴露 `build_*_instance_segmenter3d(...)` factory + `__main__` smoke
- Zoo：`dlhub/pointcloud/instance_segmentation3d_zoo.py`
  - 通过 AST 解析源码提取 `_VARIANTS` keys + build 函数名
  - Lazy import + lazy builder，避免 `import dlhub` 时一次性加载所有模型
  - arch id prefix: `pcinst3d:*`
- CLI：`scripts/instance_segmentation3d_zoo.py`
  - `--list` / `--search` / `--smoke`（随机点云 forward）
- Tests：`tests/test_dlhub_pointcloud_instance_segmentation3d_zoo.py`
  - arch 数量 >= 90（30 算法族 × 3 tiny/small/base）
  - 对所有 `*_tiny` arch 做 forward + backward smoke

---

## I/O 约定（Toy）

- **Input**: `points` tensor, shape `(B, N, C)`，其中 `C>=3`，前 3 维为 `xyz`。
- **Output**: `dict` 至少包含：
  - `mask_logits`: `(B, K, N)`，K 为实例候选数（query/prototype/pivot 等）
  - `cls_logits`: `(B, K, num_classes)`

---

## Verification Commands

- Run unit tests: `pytest -q`
- List arch ids: `python scripts/instance_segmentation3d_zoo.py --list`
- Smoke one model: `python scripts/instance_segmentation3d_zoo.py --smoke pcinst3d:mask3d_tiny`

