# Lesson 05：点云 part segmentation（PointNet, compact-first）

目标：从 “点云分类” 走到 “点级别预测（per-point classification）”：

- 输入：`(B, N, 3)` 点云
- 输出：`(B, N, C)` 每个点的类别（这里是 2 类：cube part vs sphere part）

## 运行

```bash
python -m tracks.pointcloud.lesson_05_pointnet_compact_partseg.train \
  --epochs 5 \
  --max-train-batches 50 \
  --max-eval-batches 10
```

## 代码

- 数据：复用 `tracks/pointcloud/synthetic_clouds.py` 的 `SyntheticPartSegDataset`
- 模型：`tracks/pointcloud/lesson_05_pointnet_compact_partseg/model.py`
- 训练：`tracks/pointcloud/lesson_05_pointnet_compact_partseg/train.py`
