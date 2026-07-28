# Lesson 19 — PointCloud 自监督：DINOv2（compact-first）

目标：在点云上跑通一个 **DINOv2-style** 的最小闭环（compact 版本）：

- Backbone：PointMAE-style patch tokens + Transformer（有 `[CLS]` token）
- Loss 由两部分组成：
  - **DINO**：全局 `[CLS]` 的 cross-view student/teacher distillation
  - **iBOT**：patch token 的 masked distillation（只在 masked patches 上算）

## 运行

从 repo 根目录运行：

```bash
python -m tracks.pointcloud.lesson_19_pointcloud_selfsupervised_dinov2.train --run-name dev
```

查看支持的架构变体：

```bash
python -m tracks.pointcloud.lesson_19_pointcloud_selfsupervised_dinov2.train --list-arch
```

常用可调参数：

```bash
python -m tracks.pointcloud.lesson_19_pointcloud_selfsupervised_dinov2.train --mask-ratio 0.5 --lambda-ibot 1.0
```

