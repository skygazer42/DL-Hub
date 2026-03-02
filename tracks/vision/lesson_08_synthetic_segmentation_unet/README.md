# Lesson 08 — Synthetic Segmentation (Tiny U-Net)

目标：用合成数据跑通一个最小的二分类分割闭环（U-Net 风格），并记录一致的产物与指标（loss + IoU）。

## 运行

CPU 冒烟：

```bash
python -m tracks.vision.lesson_08_synthetic_segmentation_unet.train \
  --device cpu --epochs 1 \
  --num-samples 256 --batch-size 8 \
  --max-train-batches 2 --max-eval-batches 2 \
  --run-name smoke
```

选择模型：

```bash
python -m tracks.vision.lesson_08_synthetic_segmentation_unet.train --list-arch
python -m tracks.vision.lesson_08_synthetic_segmentation_unet.train \
  --arch tvseg:fcn_resnet50 --device cpu --epochs 1 --max-train-batches 2 --max-eval-batches 2
```

输出目录：

- `outputs/vision/lesson_08_synthetic_segmentation_unet/<run_name>/config.json`
- `outputs/vision/lesson_08_synthetic_segmentation_unet/<run_name>/metrics.jsonl`
- `outputs/vision/lesson_08_synthetic_segmentation_unet/<run_name>/checkpoints/checkpoint.pt`
