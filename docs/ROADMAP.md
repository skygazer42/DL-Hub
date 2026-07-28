# 学习路线图（PyTorch 统一主线）

这个仓库被重置为 **PyTorch 主线 + 多条轨道（tracks）** 的学习项目：你可以按轨道深入，也可以按推荐顺序“从基础到进阶”系统学习。

## 推荐顺序（第一次学习）

1. **Foundations（基础）**：张量、autograd、训练循环、优化器与正则化 —— 学会“怎么训练一个模型”。
2. **Vision（视觉）**：从 MNIST/LeNet 开始，逐步学数据增强、迁移学习、解释性等通用技巧。
3. **NLP（自然语言）**：先做文本分类，再做 attention/transformer，最后做 NER/阅读理解最小实现。
4. **GNN（图神经网络）**：从 Cora 数据集入门，再到 GCN/GAT/GIN。
5. **PointCloud（点云）**：PointNet/DGCNN 最小实现，重点理解点集特性与邻域构建。
6. **Generative（生成模型）**：VAE 与 GAN（MNIST 最小实现）建立生成建模直觉。
7. **LLM（资料 + 小实验）**：保留 PDF 资料，并补充“阅读路线 + 可跑实验”。

## 轨道入口

- `tracks/foundations/`
- `tracks/vision/`
- `tracks/nlp/`
- `tracks/gnn/`
- `tracks/pointcloud/`
- `tracks/generative/`
- `tracks/llm/`

> 进度提示：截至 2026-02-28，`tracks/generative/`（VAE/GAN）与 `tracks/pointcloud/`（PointNet compact 分类）已可运行并纳入 `scripts/smoke_check.py` 冒烟。

## 每节课（lesson）统一长相

每个 `lesson_xxx/` 目录尽量包含：

- `README.md`：目标、先修、概念、练习、验收方式（最重要）
- `model.py`：模型（尽量短、可读）
- `data.py`：数据（支持 compact/fake 模式用于快速自检）
- `train.py`：训练入口（统一命令行参数风格）
- `eval.py`：评估入口（统一输出指标）

## 学习者的“验收”

学习不是“能跑就行”。每个 lesson 都会给出至少一种验收方式，例如：

- `python train.py --max-steps 50`（10 秒内跑通）
- `pytest -q`（练习题的 TODO 通过）
- 输出指标达到阈值（例如 MNIST accuracy > 98%）
