# 项目结构

## 顶层目录

```text
DL-Hub/
├── dlhub/              # 统一训练/评估脚手架（seed/device/训练循环/日志）
├── tracks/             # 课程内容（多轨统一结构）
│   ├── foundations/    #   基础赛道
│   ├── vision/        #   视觉赛道
│   ├── nlp/           #   NLP 赛道
│   ├── gnn/           #   GNN 赛道
│   ├── pointcloud/    #   点云赛道
│   ├── generative/    #   生成模型赛道
│   ├── llm/           #   LLM 赛道
│   └── multimodal/    #   多模态赛道
├── ml_algorithms/      # 纯 NumPy ML 算法（27+ 个经典算法）
├── optimization/       # 优化工具箱（Optimizer/Scheduler/Loss/Metric + 元启发式）
├── scripts/            # 辅助脚本（冒烟验证、Zoo CLI）
├── docs/               # MkDocs 文档源文件
├── resources/          # 保留资料（PDF 论文笔记）
├── tests/              # 126+ 测试文件
├── mkdocs.yml          # 文档配置
├── Makefile            # 常用命令入口
└── requirements*.txt   # 依赖清单
```

---

## 课程目录结构

每个 lesson 遵循统一的四文件结构：

```text
tracks/<track>/lesson_XX_<name>/
├── model.py        # 模型定义
├── data.py         # 数据加载（必须支持 --dataset fake）
├── train.py        # 训练脚本（入口）
└── README.md       # 课程说明文档
```

!!! example "示例：MNIST LeNet"

    ```text
    tracks/vision/lesson_01_mnist_lenet/
    ├── model.py        # LeNet-5 定义
    ├── data.py         # MNIST 数据加载 + fake 模式
    ├── train.py        # 训练循环
    └── README.md       # 课程说明
    ```

### 关键约定

| 文件 | 职责 | 要求 |
|------|------|------|
| `model.py` | 模型架构定义 | 纯 PyTorch `nn.Module` |
| `data.py` | 数据集和 DataLoader | 必须支持 `--dataset fake` |
| `train.py` | 训练入口脚本 | 使用 `dlhub/` 脚手架 |
| `README.md` | 课程文档 | 包含原理说明和运行方式 |

---

## 输出目录约定

所有训练输出统一存放在 `outputs/` 目录下：

```text
outputs/
└── <track>/
    └── <lesson>/
        └── <run_name>/
            ├── checkpoints/    # 模型权重
            ├── logs/           # 训练日志
            └── config.json     # 运行配置
```

!!! tip "输出路径管理"

    `dlhub/` 脚手架会自动创建和管理输出目录结构，
    开发者只需在 `train.py` 中指定 `run_name` 即可。

---

## scripts/ 辅助脚本

| 脚本 | 功能 |
|------|------|
| `smoke_check.py` | 全局冒烟测试，验证所有 lesson 的 `--dataset fake` 模式 |
| `run_lesson.py` | 统一运行单个 lesson |
| `new_lesson.py` | 脚手架生成新 lesson 模板 |
| `doctor.py` | 环境诊断工具 |
| `benchmark_cpu.py` | CPU 基准测试 |
| `*_zoo.py` | 各赛道 Zoo CLI 脚本（`vision_zoo.py`、`nlp_zoo.py` 等） |
