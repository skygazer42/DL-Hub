# 仓库结构速览（DL-Hub）

这个仓库正在被重置为一个 **PyTorch 主线 + 多轨课程（tracks）** 的学习项目：核心目标是“统一风格、统一训练脚手架、系统学习而不割裂”。

历史素材会作为参考被逐步吸收进新的课程结构；代码收敛到 `tracks/`，资料（PDF 等）集中到 `resources/`。在对应课程实现通过验收后，旧实现会被删除（PDF 资料不删、不改内容，且可在 Git 历史中追溯）。

## 顶层目录说明

- `dlhub/`：统一训练/评估脚手架（seed/device/训练循环/日志/输出目录约定等）。
- `tracks/`：课程内容（多轨统一结构：foundations/vision/nlp/gnn/pointcloud/generative/llm）。
- `docs/`：文档（路线图、规范、运行方式、FAQ）。
- `scripts/`：辅助脚本（冒烟验证、环境诊断等）。
- `resources/`：保留资料（以 PDF 为主），用于集中收纳从旧目录整理出来的内容。
  - `resources/pdfs/llms/`：LLM 相关论文与资料（以 PDF/MD 为主，重要资料保留）。

以下目录是历史素材，会被逐步重写吸收进 `tracks/`（不作为主入口）：
- `resources/pdfs/` 下的各类论文/笔记/报告等

> 注意：仓库内存在带空格的路径（多见于资料目录），在命令行中使用时需要加引号，例如：`cd "resources/pdfs/machine_learning_alg/Rumor prediction"`。

## 推荐从哪里开始

第一次学习建议：

1. 看 `docs/ROADMAP.md` 了解全局路线。
2. 从 `tracks/vision/lesson_01_mnist_lenet/` 跑通第一个 PyTorch 闭环（支持 `--dataset fake` 离线冒烟）。
3. 再进入 `tracks/foundations/` 补齐训练循环与理论基础（会逐步完善）。
