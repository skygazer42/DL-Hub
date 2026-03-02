# Lesson 04：PointCloud Zoo（toy 点云分类）

目标：用统一入口在 **同一份 toy 数据** 上快速切换 30+ 主流点云 backbone，跑通训练闭环并便于对比。

## 列出支持的架构

```bash
python -m tracks.pointcloud.lesson_04_pointcloud_zoo_toy_classification.train --list-arch
```

## 运行方式

快速冒烟（CPU）：

```bash
python -m tracks.pointcloud.lesson_04_pointcloud_zoo_toy_classification.train \\
  --arch pointnet --epochs 1 \\
  --max-train-batches 2 --max-eval-batches 1 \\
  --device cpu --run-name smoke
```

切换不同架构（示例）：

```bash
python -m tracks.pointcloud.lesson_04_pointcloud_zoo_toy_classification.train --arch pc:point_transformer --device cpu
python -m tracks.pointcloud.lesson_04_pointcloud_zoo_toy_classification.train --arch pc:pvcnn --device cpu
python -m tracks.pointcloud.lesson_04_pointcloud_zoo_toy_classification.train --arch pc:pointnet_tnet --device cpu
```

## 输出产物（统一规范）

`outputs/pointcloud/lesson_04_pointcloud_zoo_toy_classification/<run_name>/`

- `config.json`
- `metrics.jsonl`
- `logs/train.log`
- `checkpoints/checkpoint.pt`

