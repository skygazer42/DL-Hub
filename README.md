# DL-Hub（PyTorch 统一学习项目）

这个仓库正在被重置为一个 **PyTorch 主线 + 多轨课程（tracks）** 的学习项目：统一代码风格、统一训练/评估脚手架、统一运行方式，让学习者真正能“循序渐进跑通 → 改得动 → 能验收”。

> 重要：仓库内的 PDF 资料（例如 `resources/pdfs/llms/`）会完整保留；非 PDF 的旧实现会被逐步重写吸收进新的课程结构，等新实现验收通过后再删除旧版本（Git 历史可追溯）。

## 快速开始（2 分钟跑通）

1) 冒烟验证（不下载数据）：

```bash
python scripts/smoke_check.py
```

2) 跑通第一个 PyTorch lesson（fake 模式，不下载 MNIST）：

```bash
python -m tracks.vision.lesson_01_mnist_lenet.train --dataset fake --epochs 1 --max-train-batches 2 --max-eval-batches 2
```

## 你应该从哪里开始

- 路线图：`docs/ROADMAP.md`
- 仓库结构：`docs/STRUCTURE.md`
- 如何运行：`docs/RUNNING.md`
- 代码规范：`docs/STYLEGUIDE.md`
- 运行约定：`docs/CONVENTIONS.md`

## Tracks（课程轨道）

已落地（可运行）：

- `tracks/foundations/`：基础（张量、autograd、训练循环入门）
- `tracks/vision/`：视觉（MNIST 入门闭环）
- `tracks/gnn/`：图神经网络（toy 图任务 → Cora：GCN/LP/GraphSAGE 最小实现）
- `tracks/nlp/`：NLP（toy 文本分类 → 逐步走向 transformer/NER/阅读理解）
- `tracks/llm/`：LLM（toy causal LM → 生成 → 记录产物）
- `tracks/generative/`：生成模型（VAE/GAN 最小实现，支持 `--dataset fake` 离线冒烟）
- `tracks/pointcloud/`：点云（PointNet toy 分类，后续扩展到 DGCNN/PointNet2）

规划中（会逐步补齐到同一套脚手架）：

（暂无）

## 旧目录（正在迁移重写，不作为主入口）

历史资料已集中到 `resources/`，会作为参考素材逐步吸收进 `tracks/`，不再作为主入口：

- `resources/pdfs/` 下的论文/笔记/报告等
