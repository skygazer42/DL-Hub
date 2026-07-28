# GNN 轨（图神经网络）

目标：用**最少依赖**（尽量不依赖 PyG）把图神经网络的核心直觉跑通：邻居聚合、归一化、图级池化、训练与评估。

这条轨道优先提供 **synthetic** 数据集，让你在不下载任何数据的情况下先学会：

- 图数据如何表示（邻接矩阵 / edge list）
- GCN/GIN 等消息传递的最小实现
- 图级分类（Graph Classification）的训练闭环

## Lessons

- `lesson_01_compact_graph_classification/`：compact 图级分类（Cycle vs Star），GCN 最小实现
- `lesson_02_gin_compact_graph_classification/`：compact 图级分类（GIN 最小实现，纯 PyTorch）
- `lesson_03_gat_compact_graph_classification/`：compact 图级分类（GAT 最小实现，纯 PyTorch）
- `lesson_04_cora_node_classification_gcn/`：Cora 节点分类（GCN 最小实现，纯 PyTorch，含稀疏邻接）
- `lesson_05_label_propagation_cora/`：Cora 传统方法基线（Label Propagation，纯 PyTorch sparse）
- `lesson_06_graphsage_cora/`：Cora 节点分类（GraphSAGE 最小实现，全图训练，纯 PyTorch sparse）
- `lesson_07_sdne_karate_embedding/`：Karate 节点表示学习（SDNE 风格，自编码器 + 平滑项）
- `lesson_08_line_karate_embedding/`：Karate 节点表示学习（LINE 风格，负采样）
- `lesson_09_metapath2vec_compact_hetero_embedding/`：compact 异构图上的 metapath2vec 风格 embedding（含同类型负采样）
- `lesson_10_pinsage_compact_recommender/`：compact 推荐图上的 PinSAGE 风格 item embedding（随机游走邻居 + GraphSAGE 聚合）
- `lesson_11_rgcn_compact_node_classification/`：compact 关系图上的 R-GCN 节点分类（纯 PyTorch，含 basis 分解）
