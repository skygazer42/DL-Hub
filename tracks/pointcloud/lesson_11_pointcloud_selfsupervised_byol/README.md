# Lesson 11 — PointCloud 自监督：BYOL（toy-first）

目标：在点云上跑通 **BYOL**（Bootstrap Your Own Latent）的最小闭环：

- 两个增强视图 `v1/v2`
- Online 网络（encoder + projector + predictor）
- Target 网络（encoder + projector），用 EMA 更新
- 损失：对齐 online predictor 与 target projector 的表征（cosine similarity）

## 运行

从 repo 根目录运行：

```bash
python -m tracks.pointcloud.lesson_11_pointcloud_selfsupervised_byol.train --run-name dev
```

查看支持的架构变体：

```bash
python -m tracks.pointcloud.lesson_11_pointcloud_selfsupervised_byol.train --list-arch
```

## 你应该看到

- loss 能稳定下降（toy 数据不追求极致，只要能训练、能反向传播）
- `runs/pointcloud/lesson_11_pointcloud_selfsupervised_byol/<run-name>/` 下有 `config.json`、`metrics.jsonl` 和 checkpoint

