# Lesson 11：实例分割（Synthetic, compact-first, YOLACT-style）

目标：用一个最小闭环跑通 **实例分割** 的关键结构：**prototypes + coefficients**（YOLACT 思路）。

本 lesson 的简化点：
- 每张图只有 **1 个矩形实例**（便于理解）
- `num_anchors=1`，把输出张量形状压到最简单
- 训练只在 “中心 cell” 做分配（类似 compact FCOS）

## 运行

```bash
python -m tracks.vision.lesson_11_synthetic_instance_segmentation_yolact.train \
  --arch yolact_tiny \
  --epochs 5 \
  --max-train-batches 50 \
  --max-eval-batches 10
```

## 代码

- 数据：`tracks/vision/lesson_11_synthetic_instance_segmentation_yolact/data.py`
- 模型：`tracks/vision/lesson_11_synthetic_instance_segmentation_yolact/model.py`
- 训练：`tracks/vision/lesson_11_synthetic_instance_segmentation_yolact/train.py`
- 算法实现：`dlhub/vision/instance_segmentation/yolact.py`

