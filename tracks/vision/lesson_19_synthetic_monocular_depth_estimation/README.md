# Lesson 19 — Synthetic Monocular Depth Estimation

目标：用纯 PyTorch 生成带有显式深度标注的分层几何场景，完成一个最小的单目深度回归闭环。

## 任务定义

- 输入：灰度图 `(1, H, W)`，包含背景梯度与若干重叠的矩形/椭圆层
- 输出：深度图 `(1, H, W)`，数值范围在 `[near_depth, far_depth]`
- 监督：`depth` 为稠密真值，`occlusion` 标记前景覆盖区域，`layer_ids` 表示当前可见层编号
- 损失：`smooth_l1_loss`

## 运行

```bash
python -m tracks.vision.lesson_19_synthetic_monocular_depth_estimation.train \
  --device cpu --epochs 1 \
  --num-samples 128 --batch-size 8 \
  --max-train-batches 2 --max-eval-batches 2 \
  --run-name smoke
```

输出目录：

- `outputs/vision/lesson_19_synthetic_monocular_depth_estimation/<run_name>/config.json`
- `outputs/vision/lesson_19_synthetic_monocular_depth_estimation/<run_name>/metrics.jsonl`
- `outputs/vision/lesson_19_synthetic_monocular_depth_estimation/<run_name>/checkpoints/checkpoint.pt`
