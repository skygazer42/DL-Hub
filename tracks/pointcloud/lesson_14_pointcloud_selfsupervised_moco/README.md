# Lesson 14 — PointCloud 自监督：MoCo v2（toy-first）

目标：在点云上跑通 **MoCo v2** 的最小闭环：

- 两个增强视图 `v1/v2`
- Query encoder（可训练）+ Key encoder（EMA 更新）
- 负样本来自一个 **queue**（memory bank），不依赖大 batch
- 损失：InfoNCE / CrossEntropy（正样本在 index 0）

## 运行

从 repo 根目录运行：

```bash
python -m tracks.pointcloud.lesson_14_pointcloud_selfsupervised_moco.train --run-name dev
```

查看支持的架构变体：

```bash
python -m tracks.pointcloud.lesson_14_pointcloud_selfsupervised_moco.train --list-arch
```

可选：覆盖 queue size（负样本数）：

```bash
python -m tracks.pointcloud.lesson_14_pointcloud_selfsupervised_moco.train --queue-size 2048
```

