# Lesson 15 — PointCloud 自监督：SimSiam（compact-first）

目标：在点云上跑通 **SimSiam** 的最小闭环：

- 两个增强视图 `v1/v2`
- 模型：encoder + projector + predictor
- 训练：`p1` 对齐 `stopgrad(z2)`，以及对称项（不需要 queue / 动量编码器 / 负样本）

## 运行

从 repo 根目录运行：

```bash
python -m tracks.pointcloud.lesson_15_pointcloud_selfsupervised_simsiam.train --run-name dev
```

查看支持的架构变体：

```bash
python -m tracks.pointcloud.lesson_15_pointcloud_selfsupervised_simsiam.train --list-arch
```

