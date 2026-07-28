# Lesson 07：点云重建（PointNet AutoEncoder, compact-first）

目标：跑通一个最小的点云重建闭环：

- 输入：噪声点云 `noisy (B, N, 3)`
- 输出：重建点云 `pred (B, N, 3)`
- Loss：**Chamfer Distance**（集合距离，顺序不敏感）

## 运行

列出可用模型变种：

```bash
python -m tracks.pointcloud.lesson_07_pointnet_compact_reconstruction.train --list-arch
```

训练（默认 compact 数据，离线可跑）：

```bash
python -m tracks.pointcloud.lesson_07_pointnet_compact_reconstruction.train \
  --arch pointnet_ae:pointnet_ae_small \
  --epochs 10 \
  --max-train-batches 50 \
  --max-eval-batches 10
```

## 代码

- Chamfer：`dlhub/pointcloud/ops.py`
- 数据：`tracks/pointcloud/synthetic_clouds.py`（`SyntheticReconDataset`）
- 模型：`dlhub/pointcloud/reconstruction/pointnet_ae.py`
- Lesson glue：`tracks/pointcloud/lesson_07_pointnet_compact_reconstruction/*`

