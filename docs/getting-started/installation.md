# 安装指南

## 1. 克隆仓库

```bash
git clone https://github.com/skygazer42/DL-Hub.git
cd DL-Hub
```

---

## 2. 创建虚拟环境

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

=== "CPU only"

    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    ```

=== "CUDA 12.1"

    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    ```

=== "conda (CUDA 12.1)"

    ```bash
    conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
    ```

---

## 4. 安装项目依赖

```bash
pip install -r requirements.txt
```

---

## 5. 赛道专属依赖（可选）

部分赛道有额外依赖，按需安装：

```bash
# Vision 赛道额外依赖
pip install -r requirements-vision.txt
```

!!! info "按需安装"

    如果你只学习 Foundations 赛道，只需安装 `requirements.txt` 即可。
    赛道专属依赖文件命名格式为 `requirements-<track>.txt`，在仓库根目录查找。

---

## 6. 验证安装

运行 Smoke Check 脚本，确认环境配置正确：

```bash
python scripts/smoke_check.py
```

!!! success "预期输出"

    Smoke Check 会检测 Python 版本、PyTorch 是否可导入、CUDA 是否可用等，
    并输出检测结果摘要。如果全部通过，你的环境就准备好了。

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
