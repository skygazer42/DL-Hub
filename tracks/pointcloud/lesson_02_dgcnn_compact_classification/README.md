# Lesson 02：DGCNN（compact 点云分类）

这节课的目标：在 Lesson 01（PointNet）的基础上，引入 **动态图邻域（kNN）+ EdgeConv** 的核心思路：

- 局部邻域：每个点只看 k 个邻居
- Edge feature：用 `(x_i, x_j - x_i)` 表示局部几何关系
- 动态图：邻域在特征空间中随网络层更新

## 运行方式

快速冒烟（CPU，几秒内）：

```bash
python -m tracks.pointcloud.lesson_02_dgcnn_compact_classification.train --epochs 1 --max-train-batches 2 --max-eval-batches 1 --device cpu --run-name smoke
```

默认数据是 compact（Cube vs Sphere），不需要下载任何数据。

## 输出产物（统一规范）

`outputs/pointcloud/lesson_02_dgcnn_compact_classification/<run_name>/`

- `config.json`
- `metrics.jsonl`
- `logs/train.log`
- `checkpoints/checkpoint.pt`

## 练习（建议）

1. 调 `--k`（例如 5/10/20），观察性能与速度变化。
2. 把“动态邻域”关掉（只用输入坐标建一次邻域），对比差异。
3. 把 global pooling 从 max 换成 max+avg 拼接，看看是否更稳。

