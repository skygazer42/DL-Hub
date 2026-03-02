# Lesson 03 — MNIST + AlexNet（简化实现）

## 你将学到什么

- AlexNet 结构的核心组件：大卷积核 + 下采样 + dropout + 大 FC
- 为什么 AlexNet 通常在更大输入分辨率上训练（原始论文使用 224×224）
- 如何把同一个数据集通过 resize 变成“更接近 ImageNet 风格”的输入

## 运行（离线冒烟，无需下载）

```bash
python -m tracks.vision.lesson_03_mnist_alexnet.train --dataset fake --epochs 1 --max-train-batches 1 --max-eval-batches 1 --device cpu --run-name smoke
```

## 运行（真实 MNIST，会下载数据）

```bash
python -m tracks.vision.lesson_03_mnist_alexnet.train --dataset mnist --epochs 1 --max-train-batches 5 --max-eval-batches 2
```

## 提示

- 默认会把输入 resize 到 `--resize-to 224`，这会更慢；你可以先用 `--resize-to 64` 快速验证闭环。
