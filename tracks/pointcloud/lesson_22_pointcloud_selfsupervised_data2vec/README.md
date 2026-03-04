# Lesson 22 — PointCloud 自监督：data2vec（toy-first）

目标：在点云上跑通一个 **data2vec-style** 的最小闭环（toy 版本）：

- Backbone：PointMAE-style patch tokens + Transformer（有 `[CLS]` token）
- Student 输入 **masked patches**，Teacher 输入 **unmasked**（EMA 更新）
- Loss：回归 teacher 的表示（representation regression）
  - `CLS` token 回归
  - `masked patches` token 回归

## 运行

从 repo 根目录运行：

```bash
python -m tracks.pointcloud.lesson_22_pointcloud_selfsupervised_data2vec.train --run-name dev
```

查看支持的架构变体：

```bash
python -m tracks.pointcloud.lesson_22_pointcloud_selfsupervised_data2vec.train --list-arch
```

常用可调参数：

```bash
python -m tracks.pointcloud.lesson_22_pointcloud_selfsupervised_data2vec.train --mask-ratio 0.5 --ema-decay 0.996
```

