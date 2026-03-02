# Lesson 05：Label Propagation（Cora，传统图方法基线）

这节课的目标：用 **最少依赖、纯 PyTorch（sparse）** 跑通一个经典的半监督图学习基线：**Label Propagation**。

它的意义在于：

- 在不训练任何神经网络参数的情况下，只用“图结构 + 少量标注”传播标签。
- 作为 GCN / GraphSAGE / GAT 的对照：如果你的 GNN 还不如 LP，那么你的实现/训练通常有问题。

## 你会学到什么

- Cora 图数据的基本形态：节点、边、稀疏邻接矩阵（sparse adjacency）
- 行归一化邻接矩阵 `D^{-1}A` 的直觉（“邻居平均”）
- 迭代式传播的稳定性：`alpha`、迭代层数与“固定已标注节点”

## 运行方式

从仓库根目录运行（必须用 `-m`）：

```bash
python -m tracks.gnn.lesson_05_label_propagation_cora.train --num-layers 10 --alpha 0.9
```

快速冒烟（CPU，几秒内）：

```bash
python -m tracks.gnn.lesson_05_label_propagation_cora.train --num-layers 3 --alpha 0.9 --device cpu --run-name smoke
```

## 输出产物（统一规范）

产物目录：

`outputs/gnn/lesson_05_label_propagation_cora/<run_name>/`

包含：

- `config.json`
- `metrics.jsonl`：每次传播迭代的 train/val/test accuracy
- `preds.pt`：最终每个节点的预测分布（`(N, C)`）
- `checkpoints/checkpoint.pt`：为了统一规范而保存的 checkpoint（模型无可训练参数，state_dict 为空）

## 练习（建议）

1. 把 `alpha` 从 0.1 → 0.9 扫一遍，看 test accuracy 的变化。
2. 试试“**不固定**已标注节点”（把训练节点也允许更新），观察会发生什么。
3. 把传播用的邻接矩阵从 `D^{-1}A` 换成 `D^{-1/2}AD^{-1/2}`，比较差异。

