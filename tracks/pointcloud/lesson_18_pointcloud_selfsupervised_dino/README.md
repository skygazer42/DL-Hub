# Lesson 18 — PointCloud 自监督：DINO（compact-first）

目标：在点云上跑通 **DINO** 的最小闭环：

- 两个增强视图 `v1/v2`
- Student 网络（可训练）与 Teacher 网络（EMA 更新）
- Teacher 输出做 **centering + sharpening**（temperature）
- Student 去拟合 Teacher 的 soft targets（跨 view）

## 运行

从 repo 根目录运行：

```bash
python -m tracks.pointcloud.lesson_18_pointcloud_selfsupervised_dino.train --run-name dev
```

查看支持的架构变体：

```bash
python -m tracks.pointcloud.lesson_18_pointcloud_selfsupervised_dino.train --list-arch
```

可选：调整温度/EMA（compact 实验建议先用默认）：

```bash
python -m tracks.pointcloud.lesson_18_pointcloud_selfsupervised_dino.train --teacher-temperature 0.04 --ema-decay 0.996
```

