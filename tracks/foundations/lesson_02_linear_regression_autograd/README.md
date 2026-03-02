# Lesson 02 — Linear Regression + Autograd（最小训练闭环）

目标：用最简单的回归任务把“训练”这件事彻底搞懂：

- 数据如何进入模型
- loss 如何计算
- 反向传播如何得到梯度
- optimizer 如何更新参数
- 为什么设置 seed 能复现

## 运行（冒烟）

```bash
python -m tracks.foundations.lesson_02_linear_regression_autograd.train --epochs 1 --max-train-batches 5 --max-eval-batches 2 --device cpu --run-name smoke
```

## 练习（建议）

1. 把 `noise_std` 从 0.1 改到 1.0，观察最优 loss 能达到多少。
2. 把优化器从 SGD 换成 Adam，对比收敛速度。
3. 加入 L2 正则（weight decay），观察对泛化的影响（可以用 val split）。

