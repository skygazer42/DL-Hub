# Lesson 16 — PointCloud 自监督：SwAV（compact-first）

目标：在点云上跑通 **SwAV** 的最小闭环：

- 两个增强视图 `v1/v2`
- 模型：encoder + projector + prototypes
- 用 **Sinkhorn-Knopp** 做 balanced assignment（软聚类）
- 交换监督：`q(v1)` 去监督 `p(v2)`，以及对称项

## 运行

从 repo 根目录运行：

```bash
python -m tracks.pointcloud.lesson_16_pointcloud_selfsupervised_swav.train --run-name dev
```

查看支持的架构变体：

```bash
python -m tracks.pointcloud.lesson_16_pointcloud_selfsupervised_swav.train --list-arch
```

调原型数（compact 实验可用更小的 prototypes）：

```bash
python -m tracks.pointcloud.lesson_16_pointcloud_selfsupervised_swav.train --num-prototypes 64
```

