# 环境要求

## 软件依赖

| 依赖 | 最低版本 | 说明 |
|------|---------|------|
| **Python** | 3.10+ | 项目使用了 `match/case`、`X \| Y` 类型语法等 3.10 特性 |
| **PyTorch** | 2.0+ | 核心深度学习框架 |
| **NumPy** | 1.24+ | 数值计算基础库 |

!!! note "PyTorch 安装"

    PyTorch 的安装方式因平台和 CUDA 版本而异，请参考
    [PyTorch 官方安装指南](https://pytorch.org/get-started/locally/) 选择合适的命令。

---

## 赛道前置知识

不同赛道对数学与编程基础的要求不同，下表供参考：

| 赛道 | 前置赛道 | 数学 / 领域知识 |
|------|---------|----------------|
| **Foundations** | — | Python 基础、线性代数入门 |
| **Vision** | Foundations | 卷积运算直觉、图像基础 |
| **NLP** | Foundations | 概率论基础、文本处理基础 |
| **GNN** | Foundations | 图论基础、邻接矩阵 |
| **Point Cloud** | Foundations, Vision | 三维几何、点集处理 |
| **Generative** | Foundations, Vision | 概率分布、KL 散度、博弈论直觉 |
| **LLM** | Foundations, NLP | Transformer 原理、注意力机制 |
| **Multimodal** | Vision, NLP | 注意力机制、跨模态对齐、对比学习 |

!!! tip "不必全部掌握"

    前置知识仅供参考。每节课都包含必要的背景说明，你可以在学习过程中逐步补齐。

---

## 硬件要求

| 组件 | 最低要求 | 推荐配置 |
|------|---------|---------|
| **CPU** | 任意现代 x86_64 / ARM64 | 4 核以上 |
| **内存** | 4 GB | 8 GB+ |
| **GPU** | 不需要 | NVIDIA GPU + CUDA（加速训练） |
| **磁盘** | 1 GB（仓库 + 依赖） | 10 GB+（含数据集） |

!!! info "CPU 足够运行所有课程"

    所有课程均以 **纯 CPU 可运行** 为设计目标。GPU 仅在需要加速训练时使用。
    可选真实数据的课程使用 `--dataset fake`，其余课程默认使用内置数据完成快速验证。

---

## 操作系统

| 操作系统 | 支持状态 |
|---------|---------|
| **Linux** (Ubuntu 20.04+) | :material-check-circle: 完全支持 |
| **macOS** (12 Monterey+) | :material-check-circle: 完全支持 |
| **Windows** (10/11) | :material-check-circle: 完全支持 |

!!! note "Windows 用户"

    建议使用 WSL2 或 Anaconda 环境以获得最佳体验。
