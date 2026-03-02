# Lesson 11：R-GCN（toy 关系图，节点分类）

R-GCN（Relational GCN）用于带**多种关系类型**的图（常见于知识图谱）。它的核心点在于：

- 对每一种关系 `r`，使用一套参数 `W_r`
- 消息传递时根据边的 `rel_type` 选择对应的 `W_r`
- 为了控制参数量，可以用 **basis decomposition**：`W_r = sum_b a_{r,b} V_b`

本 lesson 用纯 PyTorch 在一个 toy 关系图上跑通节点分类闭环（不依赖 DGL / PyG）。

## 运行方式

快速冒烟（CPU，几秒内）：

```bash
python -m tracks.gnn.lesson_11_rgcn_toy_node_classification.train --device cpu --epochs 10 --run-name smoke
```

## 输出产物（统一规范）

`outputs/gnn/lesson_11_rgcn_toy_node_classification/<run_name>/`

- `config.json`
- `metrics.jsonl`
- `logs/train.log`
- `checkpoints/checkpoint.pt`

