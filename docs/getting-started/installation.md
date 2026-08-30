# 安装指南

## 1. 克隆仓库

```bash
git clone https://github.com/skygazer42/DL-Hub.git
cd DL-Hub
```

---

## 2. 创建虚拟环境

`pyproject.toml` 声明的最低版本是 Python 3.10。先确认当前解释器满足要求：

```bash
python --version
```

=== "pip + venv"

    ```bash
    python -m venv .venv
    source .venv/bin/activate   # Linux / macOS
    # .venv\Scripts\activate    # Windows
    ```

=== "conda"

    ```bash
    conda create -n dlhub python=3.10 -y
    conda activate dlhub
    ```

---

## 3. 安装 PyTorch

PyTorch 安装命令因平台和 CUDA 版本而异，请前往
[pytorch.org/get-started](https://pytorch.org/get-started/locally/) 获取适合你环境的命令。

=== "CPU 示例"

    ```bash
    python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    ```

=== "CUDA / ROCm / macOS"

    使用官方选择器按操作系统、包管理器和实际计算平台生成命令。不要从本文复制固定的
    CUDA/ROCm 版本 URL；PyTorch 当前支持的 wheel 组合会随版本发布调整。

---

## 4. 安装项目与赛道依赖

```bash
# 只使用 NumPy 公共工具
python -m pip install -e .

# 或安装某条学习赛道（推荐）
python -m pip install -e ".[vision]"
```

!!! info "按需安装"

    `pyproject.toml` 是依赖契约的唯一来源。`requirements.txt`、
    `requirements-dev.txt`、`requirements-docs.txt` 和 `requirements-vision.txt`
    仅保留为兼容入口，分别转发到项目本体或对应 extra，不再单独维护另一份版本列表。

| Extra | 安装命令 | 用途 |
|---|---|---|
| `torch` | `python -m pip install -e ".[torch]"` | 共享 PyTorch 能力 |
| `foundations` | `python -m pip install -e ".[foundations]"` | 基础课程 |
| `vision` | `python -m pip install -e ".[vision]"` | PyTorch + torchvision + timm |
| `nlp` | `python -m pip install -e ".[nlp]"` | NLP 赛道 |
| `gnn` | `python -m pip install -e ".[gnn]"` | GNN 赛道 |
| `pointcloud` | `python -m pip install -e ".[pointcloud]"` | Point Cloud 赛道 |
| `generative` | `python -m pip install -e ".[generative]"` | PyTorch + torchvision 图片/数据能力 |
| `llm` | `python -m pip install -e ".[llm]"` | LLM 赛道 |
| `multimodal` | `python -m pip install -e ".[multimodal]"` | Multimodal 赛道 |
| `all` | `python -m pip install -e ".[all]"` | 全部运行时依赖 |
| `dev` | `python -m pip install -e ".[dev]"` | 测试、lint、覆盖率与打包工具 |
| `docs` | `python -m pip install -e ".[docs]"` | MkDocs 严格构建与本地预览 |

PyTorch 的 wheel 与 CUDA/CPU 平台有关。建议先执行第 3 步的官方平台命令，再安装 extra；
pip 会复用已经安装的兼容 PyTorch。

完整开发环境可在安装平台对应的 PyTorch 后一次装齐：

```bash
python -m pip install -e ".[all,dev,docs]"
python -m pip check
```

`requirements-dev.txt` 只转发到 `dev` extra，不包含 PyTorch 运行时；需要运行完整测试或课程时还应安装相应赛道或 `all` extra。

---

## 5. 验证安装

运行精选离线 Smoke Check，确认 NumPy 工具以及已安装 PyTorch 时的 8 赛道代表性训练链路可执行：

```bash
python scripts/smoke_check.py
```

!!! success "预期输出"

    脚本最后输出 `smoke_check: OK` 和实际覆盖数量。它不是 339 个课程的全量训练；
    可先用 `python scripts/smoke_check.py --list` 查看精选清单。

Vision 环境还可以用 `python scripts/doctor.py` 查看 Python、PyTorch、torchvision 与 CUDA/MPS 状态。

---

## 常见问题

??? question "安装 PyTorch 时报错 `No matching distribution`"

    通常是因为 Python 版本或系统架构不匹配。请确认：

    - Python 版本 >= 3.10
    - 使用的 pip 索引 URL 与你的平台匹配

??? question "`ModuleNotFoundError: No module named 'dlhub'`"

    DL-Hub 通过 Python 模块路径引用内部包。请确保你在仓库根目录运行命令，
    或者将仓库根目录加入 `PYTHONPATH`：

    ```bash
    export PYTHONPATH="$(pwd):$PYTHONPATH"
    ```

??? question "如何在没有网络的环境下使用？"

    所有课程都提供离线运行路径：可选真实数据的课程使用 `--dataset fake`，
    synthetic 课程默认使用内置数据，不需要传 `--dataset`。
