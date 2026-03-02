# Lesson 02：Transformer Encoder（toy 文本分类）

这节课的目标：在 Lesson 01（embedding mean pooling）的基础上，引入 **self-attention / transformer encoder**，
让你真正看清楚：

- Q/K/V 是怎么来的
- padding mask 是怎么影响注意力的
- encoder 输出如何做 pooling 得到句子级表示

## 运行方式

```bash
python -m tracks.nlp.lesson_02_toy_text_classification_transformer.train --epochs 3 --embed-dim 64 --num-heads 4 --num-layers 2
```

快速冒烟：

```bash
python -m tracks.nlp.lesson_02_toy_text_classification_transformer.train --epochs 1 --max-train-batches 2 --max-eval-batches 1 --device cpu --run-name smoke
```

## 输出产物（统一规范）

`outputs/nlp/lesson_02_toy_text_classification_transformer/<run_name>/`

- `config.json`
- `vocab.json`
- `metrics.jsonl`
- `logs/train.log`
- `checkpoints/checkpoint.pt`

## 练习（建议）

1. 把 pooling 从 masked mean 换成 “取第一个 token”（需要引入 `<cls>` token）。
2. 把激活函数从 ReLU 换成 GELU，观察变化。
3. 把 `num_layers`/`embed_dim` 扫一下，记录拟合速度与泛化变化（本 toy 数据很容易过拟合）。

