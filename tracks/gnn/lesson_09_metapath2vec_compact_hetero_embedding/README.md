# Lesson 09：metapath2vec（compact 异构图，节点表示学习）

这节课做一件事：在**异构图**上用 metapath 随机游走生成序列，再用 skip-gram + negative sampling 学出节点 embedding。

你会看到三个关键组件如何拼起来：

- 异构图的**类型约束随机游走**（metapath）
- 序列 → (center, context) 的 skip-gram 训练对
- 负采样（全局 or 同类型负采样）

## 运行方式

快速冒烟（CPU，几秒内）：

```bash
python -m tracks.gnn.lesson_09_metapath2vec_compact_hetero_embedding.train \
  --device cpu --epochs 2 --run-name smoke
```

指定 metapath（关系序列，用逗号分隔）：

```bash
python -m tracks.gnn.lesson_09_metapath2vec_compact_hetero_embedding.train \
  --metapath A2P,P2A --epochs 5 --run-name apa
```

## 输出产物（统一规范）

`outputs/gnn/lesson_09_metapath2vec_compact_hetero_embedding/<run_name>/`

- `config.json`
- `metrics.jsonl`
- `embeddings.pt`：包含 `u_embeddings`/`v_embeddings` 与节点元信息
- `logs/train.log`
- `checkpoints/checkpoint.pt`

