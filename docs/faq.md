# 常见问题

---

## 1. 运行报错：ModuleNotFoundError

**问题**：运行某个 lesson 时提示找不到模块。

**解决**：始终在仓库根目录运行，并确保已安装依赖。

```bash
cd DL-Hub
pip install -r requirements-dev.txt
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

**解决**：使用 `--dataset fake` 或 `--dataset toy` 模式进行离线测试。

```bash
# 所有 lesson 都支持 fake 模式
python -m tracks.vision.lesson_01_mnist_lenet.train --dataset fake --epochs 1
```

!!! info "fake 模式"

    fake 模式使用随机生成的数据，形状与真实数据一致，
    适用于验证代码逻辑、CI 测试、无网络环境。

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
4. 运行 `make lint && make test`
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

## 8. 如何只安装某个 track 的依赖？

部分赛道提供独立的 requirements 文件：

```bash
# 安装视觉赛道依赖
pip install -r tracks/vision/requirements.txt

# 安装 NLP 赛道依赖
pip install -r tracks/nlp/requirements.txt
```

!!! note "最小依赖"

    如果赛道目录下没有独立的 `requirements.txt`，
    使用根目录的 `requirements-dev.txt` 即可覆盖所有依赖。

    ML 算法部分 (`ml_algorithms/`) 只依赖 NumPy，无需额外安装。
