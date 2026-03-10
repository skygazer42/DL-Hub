# Lesson 13：合成行人检测（FCOS-style，anchor-free）

这节课的目标是用一个**完全离线**的合成数据集跑通「行人检测」的最小闭环：

- 数据：随机噪声背景 + 1 个“行人”（瘦高矩形）
- 模型：复用本仓库 detection local zoo 的 `dldet:pedestrian_fcos`
- 训练：cls + ltrb 回归 + centerness（可选）
- 指标：center-acc（中心点网格是否命中）+ mean IoU（解码 1 个 bbox）

> 说明：这是 toy-first 教学实现，不包含 NMS / 多框解码 / COCO AP 等工程化指标。

## 运行方式

快速冒烟（CPU，几秒内）：

```bash
python -m tracks.vision.lesson_13_synthetic_pedestrian_detection_fcos.train \
  --device cpu --epochs 2 --max-train-batches 5 --max-eval-batches 3 \
  --run-name smoke
```

切换 width multiplier（更小更快）：

```bash
python -m tracks.vision.lesson_13_synthetic_pedestrian_detection_fcos.train \
  --device cpu --width-mult 0.35 --epochs 2 --max-train-batches 5 --max-eval-batches 3 \
  --run-name tiny
```

## 输出产物（统一规范）

`outputs/vision/lesson_13_synthetic_pedestrian_detection_fcos/<run_name>/`

- `config.json`
- `metrics.jsonl`
- `logs/train.log`
- `checkpoints/checkpoint.pt`

