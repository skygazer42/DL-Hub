# Lesson 04：合成目标检测（FCOS 风格，anchor-free）

这节课用一个**合成数据集**跑通目标检测的最小闭环，并且刻意选择 **anchor-free**（类似 FCOS）的形式：

- 在特征图上做一个 **center 分类**（哪里是目标中心）
- 在中心位置回归 **l/t/r/b**（到边界的距离）
- 推理时：取分类得分最高的位置，把回归的距离解码成 bbox

数据是随机生成的灰度图：背景噪声 + 一个亮色矩形框，所以不需要任何下载。

## 运行方式

快速冒烟（CPU，几秒内）：

```bash
python -m tracks.vision.lesson_04_synthetic_detection_fcos.train \
  --device cpu --epochs 2 --max-train-batches 5 --max-eval-batches 5 --run-name smoke
```

## 输出产物（统一规范）

`outputs/vision/lesson_04_synthetic_detection_fcos/<run_name>/`

- `config.json`
- `metrics.jsonl`
- `logs/train.log`
- `checkpoints/checkpoint.pt`

