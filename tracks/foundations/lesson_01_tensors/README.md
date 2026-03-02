# Lesson 01 — Tensors & Shapes（张量与形状）

目标：建立对 Tensor 的 shape/dtype/device 的直觉，这是后面所有模型代码“看得懂”的前提。

## 运行

```bash
python -m tracks.foundations.lesson_01_tensors.run
```

## 练习（建议）

1. 修改 `run.py`：把 `float32` 改成 `float16`，看看输出 dtype 与误差变化。
2. 自己实现一个 `normalize(x)`：对最后一维做标准化（均值 0 方差 1）。
3. 用 `torch.einsum` 写一个矩阵乘法，并对照 `@` 结果一致。

