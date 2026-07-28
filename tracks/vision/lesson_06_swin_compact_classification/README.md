# Lesson 06：Swin 风格（compact 合成分类）

这节课实现一个简化版的 Swin 风格模型，把最关键的结构跑通：

- patch embedding 得到 2D 特征图
- window self-attention（局部窗口注意力）
- shifted window（交替 shift，形成跨窗口的信息流动）
- 全局池化 + 分类 head

数据与 ViT lesson 相同：噪声背景 + 亮色方块，label 是方块所在象限（4 类）。

## 运行方式

快速冒烟（CPU，几秒内）：

```bash
python -m tracks.vision.lesson_06_swin_compact_classification.train \
  --device cpu --epochs 3 --max-train-batches 5 --max-eval-batches 5 --run-name smoke
```

## 输出产物（统一规范）

`outputs/vision/lesson_06_swin_compact_classification/<run_name>/`

- `config.json`
- `metrics.jsonl`
- `logs/train.log`
- `checkpoints/checkpoint.pt`

