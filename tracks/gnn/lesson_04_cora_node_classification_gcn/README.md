# Lesson 04：Cora 节点分类（GCN）

本课目标：在经典数据集 **Cora** 上跑通 **半监督节点分类（transductive）** 的 GCN 训练闭环。

这节课对应仓库旧实现参考：`graph/pygcn/pygcn改/`。这里把它重写为统一的 `tracks/` lesson：

- 统一输出结构：`config.json` / `metrics.jsonl` / `checkpoints/checkpoint.pt`
- 统一 seed/device/logging 习惯
- 纯 PyTorch 实现稀疏邻接矩阵归一化（不依赖 SciPy）

## 运行

从仓库根目录运行：

```bash
python -m tracks.gnn.lesson_04_cora_node_classification_gcn.train \
  --epochs 50 --device cpu --run-name dev
```

快速冒烟：

```bash
python -m tracks.gnn.lesson_04_cora_node_classification_gcn.train \
  --epochs 1 --device cpu --run-name smoke
```

## 你将学到

- Cora 的输入结构：一个图 + 全部节点特征 + 节点标签 + 三个 index split
- GCN 的核心：`H^{l+1} = σ(Â H^l W^l)`（Â 为加自环后的归一化邻接）
- 为什么训练只用 `idx_train`，但 forward 在整张图上做（transductive）

