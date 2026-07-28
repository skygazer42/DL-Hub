# Lesson 01：PointNet（compact 点云分类）

这节课的目标：用最小实现跑通 **PointNet 分类** 的训练闭环，并理解：

- 点云输入形状：`(B, N, 3)`
- per-point MLP（用 `Conv1d` 模拟）+ **对点维度做 max pooling** 得到全局特征
- 为什么 max pooling 能提供 permutation invariance（对点顺序不敏感）

## 运行方式

快速冒烟（CPU，几秒内）：

```bash
python -m tracks.pointcloud.lesson_01_pointnet_compact_classification.train --epochs 1 --max-train-batches 2 --max-eval-batches 1 --device cpu --run-name smoke
```

默认数据是 compact（Cube vs Sphere），不需要下载任何数据。

## 输出产物（统一规范）

`outputs/pointcloud/lesson_01_pointnet_compact_classification/<run_name>/`

- `config.json`
- `metrics.jsonl`
- `logs/train.log`
- `checkpoints/checkpoint.pt`

## 练习（建议）

1. 把 `num_points` 从 64 → 512，观察训练速度与效果的变化。
2. 把 pooling 从 max 换成 mean，比较差异。
3. 给点加随机旋转增强（rotation augmentation），提高泛化。

