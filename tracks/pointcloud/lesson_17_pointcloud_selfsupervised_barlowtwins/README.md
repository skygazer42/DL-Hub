# Lesson 17 — PointCloud 自监督：Barlow Twins（compact-first）

目标：在点云上跑通 **Barlow Twins** 的最小闭环（冗余约束 / 去相关）：

- 两个增强视图 `v1/v2`
- 模型：encoder + projector
- 目标：让 cross-correlation 矩阵接近单位阵  
  - 对角项接近 1（对齐）
  - 非对角项接近 0（去冗余）

## 运行

从 repo 根目录运行：

```bash
python -m tracks.pointcloud.lesson_17_pointcloud_selfsupervised_barlowtwins.train --run-name dev
```

查看支持的架构变体：

```bash
python -m tracks.pointcloud.lesson_17_pointcloud_selfsupervised_barlowtwins.train --list-arch
```

调 off-diagonal 权重（compact 实验可用更小/更大）：

```bash
python -m tracks.pointcloud.lesson_17_pointcloud_selfsupervised_barlowtwins.train --lambda-offdiag 0.005
```

