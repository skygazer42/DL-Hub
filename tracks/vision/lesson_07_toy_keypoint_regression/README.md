# Lesson 07 — Toy Keypoint Regression

目标：用**纯合成数据**跑通一个最小的关键点回归闭环（数据 → 模型 → 训练 → 评估 → 产物），并把输出约定与其他 vision lessons 统一起来。

## 任务定义

- 输入：灰度图 `(1, H, W)`，其中包含一个亮点（高斯点）+ 噪声
- 输出：关键点坐标 `(x_norm, y_norm)`，归一化到 `[0, 1]`
- 损失：MSE
- 额外指标：`eval_l2_px`（像素空间下的平均 L2 误差）

## 运行

CPU 冒烟（不依赖下载数据）：

```bash
python -m tracks.vision.lesson_07_toy_keypoint_regression.train \
  --device cpu --epochs 1 \
  --num-samples 256 --batch-size 32 \
  --max-train-batches 2 --max-eval-batches 2 \
  --run-name smoke
```

输出目录：

- `outputs/vision/lesson_07_toy_keypoint_regression/<run_name>/config.json`
- `outputs/vision/lesson_07_toy_keypoint_regression/<run_name>/metrics.jsonl`
- `outputs/vision/lesson_07_toy_keypoint_regression/<run_name>/checkpoints/checkpoint.pt`

