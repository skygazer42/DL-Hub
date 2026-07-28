# Lesson 06：点云 part segmentation（DGCNN, compact-first）

目标：用 **动态图 kNN** 的 EdgeConv 直觉做点级别预测（per-point classification）。

数据与 lesson 05 相同：同一个点云里有两部分：
- part 0：cube
- part 1：sphere

## 运行

```bash
python -m tracks.pointcloud.lesson_06_dgcnn_compact_partseg.train \
  --epochs 5 \
  --max-train-batches 50 \
  --max-eval-batches 10
```

## 代码

- 数据：复用 `tracks/pointcloud/synthetic_clouds.py` 的 `SyntheticPartSegDataset`
- 模型：`tracks/pointcloud/lesson_06_dgcnn_compact_partseg/model.py`
- 训练：`tracks/pointcloud/lesson_06_dgcnn_compact_partseg/train.py`
