# Lesson 01 — MNIST + LeNet（入门闭环）

## 你将学到什么

- MNIST 分类任务的最小闭环：数据 → 模型 → 训练 → 评估
- 为什么 LeNet 结构适合做第一个 CNN
- 如何用一致的命令行参数快速做冒烟验证（少量 batch）

## 先修

- `tracks/foundations/` 的张量与训练循环基础（没有也能跑，但理解会慢）

## 运行（快速冒烟，无需下载）

```bash
python -m tracks.vision.lesson_01_mnist_lenet.train --dataset fake --epochs 1 --max-train-batches 2 --max-eval-batches 2
```

## 运行（真实 MNIST，会下载数据）

```bash
python -m tracks.vision.lesson_01_mnist_lenet.train --dataset mnist --epochs 1 --max-train-batches 5 --max-eval-batches 2
```

## 练习（建议）

1. 把 `LeNet` 的激活函数从 ReLU 换成 GELU，观察收敛速度变化。
2. 把优化器从 Adam 换成 SGD+Momentum，调整学习率让它也能收敛。
3. 加一个 `--weight-decay` 参数，观察过拟合趋势（可用更长训练验证）。

## 验收

- `--dataset fake` 模式下，脚本应在 10 秒内跑完并输出 train/eval 指标。
