# Lesson 21 — PointCloud 自监督：MSN（Masked Siamese Networks, compact-first）

目标：在点云上跑通一个 **MSN-style** 的最小闭环（compact 版本）：

- Backbone：PointMAE-style patch tokens + Transformer（有 `[CLS]` token）
- Student 输入 **masked patches**（更难的视图），Teacher 输入 **unmasked**（EMA 更新）
- Loss：
  - Prototype distillation：teacher→student cross-view 软标签蒸馏
  - Prototype balance：约束 student 的平均分配接近均匀，避免塌缩

## 运行

从 repo 根目录运行：

```bash
python -m tracks.pointcloud.lesson_21_pointcloud_selfsupervised_msn.train --run-name dev
```

查看支持的架构变体：

```bash
python -m tracks.pointcloud.lesson_21_pointcloud_selfsupervised_msn.train --list-arch
```

常用可调参数：

```bash
python -m tracks.pointcloud.lesson_21_pointcloud_selfsupervised_msn.train --mask-ratio 0.5 --entropy-weight 1.0
```

