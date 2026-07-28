# Lesson 10：点云自监督（PointMAE-style, compact-first）

目标：用 **Masked AutoEncoder** 思路做点云自监督预训练：

- 点云先做 patch 分组（FPS + kNN）
- mask 一部分 patch token
- Transformer encoder 只看 visible tokens
- decoder 复原 masked patch 的局部点坐标
- loss：对 masked patch 做 **Chamfer Distance**

## 运行

列出可用变种：

```bash
python -m tracks.pointcloud.lesson_10_pointcloud_selfsupervised_pointmae.train --list-arch
```

训练（建议先 smoke）：

```bash
python -m tracks.pointcloud.lesson_10_pointcloud_selfsupervised_pointmae.train \
  --arch pointmae:pointmae_tiny \
  --epochs 10 \
  --mask-ratio 0.6 \
  --max-train-batches 100 \
  --max-eval-batches 20
```

## 代码

- 模型：`dlhub/pointcloud/selfsupervised/pointmae.py`
- 数据：`tracks/pointcloud/lesson_10_pointcloud_selfsupervised_pointmae/data.py`
- 训练：`tracks/pointcloud/lesson_10_pointcloud_selfsupervised_pointmae/train.py`

