# Lesson 02 — MNIST + MLP（全连接基线）

## 你将学到什么

- 为什么在图像任务中 MLP 往往不如 CNN（但非常适合做基线）
- 如何把图像 flatten 成向量输入全连接网络
- 如何用统一脚手架快速做离线冒烟验证

## 运行（离线冒烟，无需下载）

```bash
python -m tracks.vision.lesson_02_mnist_mlp.train --dataset fake --epochs 1 --max-train-batches 2 --max-eval-batches 2
```

## 运行（真实 MNIST，会下载数据）

```bash
python -m tracks.vision.lesson_02_mnist_mlp.train --dataset mnist --epochs 1 --max-train-batches 5 --max-eval-batches 2
```

## 练习（建议）

1. 把隐藏层从 300 改到 128/512，观察训练速度与效果变化。
2. 加入第二个隐藏层（两层 MLP），观察是否过拟合。
3. 加入 Dropout，观察泛化变化。

