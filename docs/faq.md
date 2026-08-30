# 常见问题

---

## 1. 运行报错：ModuleNotFoundError

**问题**：运行某个 lesson 时提示找不到模块。

**解决**：始终在仓库根目录运行，并确保已安装依赖。

```bash
cd DL-Hub
python -m pip install -e ".[vision]"
python -m tracks.vision.lesson_01_mnist_lenet.train --dataset fake
```

!!! tip "快速排查"

    如果你在子目录中直接运行脚本，Python 找不到 `dlhub` 等包。
    务必回到仓库根目录，或将根目录加到 `PYTHONPATH`：

    ```bash
    export PYTHONPATH=/path/to/DL-Hub:$PYTHONPATH
    ```

---

## 2. CUDA 不可用 / 训练很慢

**问题**：提示 CUDA 不可用，或者训练速度很慢。

**解决**：

1. 先使用 CPU 确认代码能正常运行：

    ```bash
    python -m tracks.vision.lesson_01_mnist_lenet.train --device cpu --dataset fake
    ```

2. 检查 GPU 状态：

    ```bash
    nvidia-smi
    python -c "import torch; print(torch.cuda.is_available())"
    ```

3. 确认 PyTorch 安装了 CUDA 版本（参考 [PyTorch 官网](https://pytorch.org/get-started/locally/)）

---

## 3. 数据下载失败

**问题**：网络问题导致数据集下载失败或超时。

**解决**：可选真实数据的课程使用 `--dataset fake`；synthetic 课程默认使用内置数据。

```bash
# MNIST 等可选真实数据课程支持 fake 模式
python -m tracks.vision.lesson_01_mnist_lenet.train --dataset fake --epochs 1
```

!!! info "fake 模式"

    fake 模式使用随机生成的数据，形状与真实数据一致。没有 `--dataset` 参数的课程
    已经使用内置离线数据，可先运行 `run_lesson.py ... --describe` 查看实际能力。

---

## 4. 为什么不用 PyTorch Lightning / HuggingFace Trainer？

DL-Hub 是一个**学习项目**，核心目标是让你理解训练过程的每一个细节。

- 使用**最少抽象**，训练循环完全可见
- `dlhub/` 脚手架只提供种子固定、设备管理、路径管理等基础功能
- 鼓励你阅读和修改训练循环代码

> "Tell me and I forget, teach me and I may remember, involve me and I learn."

---

## 5. 如何选择学习路线？

推荐三条入门路线：

| 路线 | 适合人群 | 起点 |
|------|----------|------|
| **视觉优先** | 对图像处理感兴趣 | `tracks/vision/lesson_01_mnist_lenet/` |
| **NLP 优先** | 对文本处理感兴趣 | `tracks/nlp/` |
| **理论优先** | 希望先打基础 | `tracks/foundations/` |

详情参见 [学习赛道总览](tracks/index.md)。

---

## 6. 如何贡献？

参见 [贡献指南](developer/contributing.md)，完整流程：

1. Fork 仓库
2. 创建功能分支
3. 编写代码和测试
4. 运行 `make check`，并按改动范围补充 `make docs` 或 `make smoke`
5. 提交 Pull Request

---

## 7. Zoo 模型如何使用？

每个赛道的 Zoo 提供统一的 CLI 接口：

```bash
# 列出所有可用模型
python scripts/vision_zoo.py --list

# 搜索特定模型
python scripts/vision_zoo.py --search resnet

# 冒烟测试所有模型
python scripts/vision_zoo.py --smoke
```

!!! example "Detection Zoo 示例"

    ```bash
    # 列出所有检测模型
    python scripts/detection_zoo.py --list

    # 搜索行人检测 preset
    python scripts/detection_zoo.py --search pedestrian

    # 冒烟测试
    python scripts/detection_zoo.py --smoke
    ```

---

## 8. 如何按赛道安装依赖？

依赖统一由根目录 `pyproject.toml` 的 extras 管理：

```bash
# 安装基础工具
python -m pip install -e .

# 按赛道安装
python -m pip install -e ".[vision]"
python -m pip install -e ".[nlp]"
python -m pip install -e ".[gnn]"

# 安装全部运行时依赖
python -m pip install -e ".[all]"
```

!!! note "当前约定"

    `requirements-vision.txt` 只为旧命令兼容而保留，内部转向 `.[vision]`。
    PyTorch 的 CPU/CUDA wheel 请先按[安装指南](getting-started/installation.md)选择；
    ML 算法部分 (`ml_algorithms/`) 只依赖 NumPy。
