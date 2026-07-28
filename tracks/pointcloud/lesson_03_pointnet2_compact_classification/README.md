# Lesson 03：PointNet2（compact 点云分类）

这节课的目标：在 PointNet 的基础上引入“**分层采样 + 局部聚合**”的思路：

- 用 FPS（farthest point sampling）选出代表点（centroids）
- 每个 centroid 聚合 k 个近邻（kNN）
- 对每个局部邻域做 PointNet-style 的 MLP + pooling
- 堆叠两层 SA（set abstraction），得到更强的局部到全局表示

## 运行方式

快速冒烟（CPU，几秒内）：

```bash
python -m tracks.pointcloud.lesson_03_pointnet2_compact_classification.train --epochs 1 --max-train-batches 2 --max-eval-batches 1 --device cpu --run-name smoke
```

默认数据是 compact（Cube vs Sphere），不需要下载任何数据。

## 输出产物（统一规范）

`outputs/pointcloud/lesson_03_pointnet2_compact_classification/<run_name>/`

- `config.json`
- `metrics.jsonl`
- `logs/train.log`
- `checkpoints/checkpoint.pt`

## 练习（建议）

1. 把 `--npoint1` / `--k1` / `--k2` 扫一遍，观察速度与效果变化。
2. 把 FPS 换成随机采样，看看局部覆盖的影响。
3. 给点云加旋转增强，提高泛化。

