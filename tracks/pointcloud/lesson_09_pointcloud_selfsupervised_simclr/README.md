# Lesson 09：点云自监督（SimCLR, compact-first）

目标：用最小闭环跑通 **对比学习（SimCLR）** 的核心流程：

- 同一个点云做两次增强 → 得到两份 view
- 编码器 + projection head
- loss：NT-Xent（InfoNCE）

这节课使用 compact 数据（Cube/Sphere 混合），训练时 **不使用标签**；标签仅用于你后续扩展做 linear probe。

## 运行

列出可用模型变种：

```bash
python -m tracks.pointcloud.lesson_09_pointcloud_selfsupervised_simclr.train --list-arch
```

训练：

```bash
python -m tracks.pointcloud.lesson_09_pointcloud_selfsupervised_simclr.train \
  --arch simclr_pointnet:simclr_pointnet_small \
  --epochs 20 \
  --max-train-batches 100 \
  --max-eval-batches 20
```

## 代码

- 模型：`dlhub/pointcloud/selfsupervised/simclr.py`
- 数据增强：`tracks/pointcloud/lesson_09_pointcloud_selfsupervised_simclr/data.py`
- 训练：`tracks/pointcloud/lesson_09_pointcloud_selfsupervised_simclr/train.py`

