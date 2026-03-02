# Lesson 07：SDNE（Karate，节点表示学习）

这节课的目标：用一个小图（Karate Club）跑通 **SDNE 风格**的节点表示学习闭环：

- 输入是邻接矩阵的一行（每个节点的邻居指示）
- 通过自编码器（encoder/decoder）学习低维 embedding
- 通过边上的平滑正则项，让相连节点的 embedding 更接近

它的价值在于：把“图 embedding”这件事从 GCN/GAT 的神经消息传递中拆出来单独理解。

## 运行方式

快速冒烟（CPU，几秒内）：

```bash
python -m tracks.gnn.lesson_07_sdne_karate_embedding.train --epochs 5 --device cpu --run-name smoke
```

## 输出产物（统一规范）

`outputs/gnn/lesson_07_sdne_karate_embedding/<run_name>/`

- `config.json`
- `metrics.jsonl`
- `embeddings.pt`：`(N, D)` 的节点 embedding
- `logs/train.log`
- `checkpoints/checkpoint.pt`

## 练习（建议）

1. 调 `--lambda-smooth` 看 embedding 的“平滑程度”如何变化。
2. 把 decoder 输出改成对称重建（例如 `Z Z^T`），对比差异。
3. 用 `ml_algorithms/python/kmeans.py` 在 embedding 上做聚类，看看是否形成合理社区。

