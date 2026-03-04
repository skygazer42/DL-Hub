# Lesson 13 — PointCloud：自监督表征的 Linear Probe / Fine-tune（toy-first）

目标：把前面自监督（SimCLR / BYOL / VICReg）学到的表征，接一个 **线性分类头** 做 toy 分类（Cube vs Sphere）。

你可以用两种模式：

- **Linear probe**（默认）：冻结 SSL backbone，只训练线性头
- **Fine-tune**：解冻 SSL backbone + 线性头一起训练

## 运行

从 repo 根目录运行：

```bash
python -m tracks.pointcloud.lesson_13_pointcloud_ssl_linear_probe.train --run-name dev
```

列出支持的 SSL 架构 ID：

```bash
python -m tracks.pointcloud.lesson_13_pointcloud_ssl_linear_probe.train --list-ssl-arch
```

## 使用预训练 checkpoint

例如：先跑完 lesson 09 SimCLR，再做 probe：

```bash
python -m tracks.pointcloud.lesson_09_pointcloud_selfsupervised_simclr.train --run-name simclr_dev
python -m tracks.pointcloud.lesson_13_pointcloud_ssl_linear_probe.train \
  --run-name probe_simclr \
  --ssl-arch simclr_pointnet:simclr_pointnet_small \
  --ssl-checkpoint runs/pointcloud/lesson_09_pointcloud_selfsupervised_simclr/simclr_dev/checkpoints/checkpoint.pt
```

Fine-tune 模式（解冻 backbone）：

```bash
python -m tracks.pointcloud.lesson_13_pointcloud_ssl_linear_probe.train --freeze-ssl 0
```

