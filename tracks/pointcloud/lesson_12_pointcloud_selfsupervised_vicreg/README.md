# Lesson 12 — PointCloud 自监督：VICReg（compact-first）

目标：在点云上跑通 **VICReg** 的最小闭环（不需要负样本队列）：

- 两个增强视图 `v1/v2`
- 模型：PointNet encoder + projector
- 损失由三部分组成：
  - Invariance：`z1` 与 `z2` 的 MSE
  - Variance：每个维度 std 不要塌缩到 0
  - Covariance：不同维度不要高度相关（去相关）

## 运行

从 repo 根目录运行：

```bash
python -m tracks.pointcloud.lesson_12_pointcloud_selfsupervised_vicreg.train --run-name dev
```

查看支持的架构变体：

```bash
python -m tracks.pointcloud.lesson_12_pointcloud_selfsupervised_vicreg.train --list-arch
```

## 你应该看到

- loss 能稳定下降（compact 数据不追求极致，只要能训练、能反向传播）
- `runs/pointcloud/lesson_12_pointcloud_selfsupervised_vicreg/<run-name>/` 下有 `config.json`、`metrics.jsonl` 和 checkpoint

