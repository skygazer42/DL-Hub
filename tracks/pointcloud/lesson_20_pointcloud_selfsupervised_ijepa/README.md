# Lesson 20 — PointCloud 自监督：I-JEPA（compact-first）

目标：在点云上跑通一个 **I-JEPA-style** 的最小闭环（compact 版本）：

- Student：带 mask token 的 patch transformer + predictor
- Teacher：无 mask 的 patch transformer（EMA 更新）
- Loss：只在 **masked patches** 上，让 student 预测 teacher 的 patch embedding（cosine loss）

## 运行

从 repo 根目录运行：

```bash
python -m tracks.pointcloud.lesson_20_pointcloud_selfsupervised_ijepa.train --run-name dev
```

查看支持的架构变体：

```bash
python -m tracks.pointcloud.lesson_20_pointcloud_selfsupervised_ijepa.train --list-arch
```

可调参数：

```bash
python -m tracks.pointcloud.lesson_20_pointcloud_selfsupervised_ijepa.train --mask-ratio 0.5 --ema-decay 0.996
```

