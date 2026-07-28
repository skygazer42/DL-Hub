# Lesson 05：ViT（compact 合成分类）

这节课用一个很小的合成图像分类任务，把 **Vision Transformer** 的核心组件跑通：

- patch embedding（把图像切成 patch 并映射到 token）
- Transformer encoder blocks
- CLS token + 分类 head

数据不需要下载：图片是噪声背景 + 一个亮色方块，label 是方块所在象限（4 类）。

## 运行方式

快速冒烟（CPU，几秒内）：

```bash
python -m tracks.vision.lesson_05_vit_compact_classification.train \
  --device cpu --epochs 3 --max-train-batches 5 --max-eval-batches 5 --run-name smoke
```

## 输出产物（统一规范）

`outputs/vision/lesson_05_vit_compact_classification/<run_name>/`

- `config.json`
- `metrics.jsonl`
- `logs/train.log`
- `checkpoints/checkpoint.pt`

