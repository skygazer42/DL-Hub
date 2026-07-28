# Lesson 23 — PointCloud 自监督：ReSSL（Relational Self-Supervised Learning, compact-first）

目标：在点云上跑通一个 **ReSSL-style** 的最小闭环（compact 版本）：

- Student / Teacher（EMA）双网络
- Student 看 “strong augmentation”，Teacher 看 “weak augmentation”
- 用 **关系蒸馏（relational distillation）** 学习：让 student 的相似度分布去匹配 teacher 的相似度分布
- 引入 **queue** 作为 memory bank，增强负样本/关系对数量

## 运行

从 repo 根目录运行：

```bash
python -m tracks.pointcloud.lesson_23_pointcloud_selfsupervised_ressl.train --run-name dev
```

查看支持的架构变体：

```bash
python -m tracks.pointcloud.lesson_23_pointcloud_selfsupervised_ressl.train --list-arch
```

常用可调参数：

```bash
python -m tracks.pointcloud.lesson_23_pointcloud_selfsupervised_ressl.train \
  --queue-size 1024 --ema-decay 0.99 \
  --student-temperature 0.2 --teacher-temperature 0.04
```

