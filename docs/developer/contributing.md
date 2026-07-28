# 贡献指南

感谢你对 DL-Hub 的关注！以下是参与贡献的完整流程。

---

## 贡献流程

```mermaid
graph LR
    A[Fork 仓库] --> B[创建分支]
    B --> C[编写代码]
    C --> D[运行测试]
    D --> E[提交 PR]
```

### 1. Fork 与克隆

```bash
git clone https://github.com/<your-username>/DL-Hub.git
cd DL-Hub
pip install -r requirements-dev.txt
```

### 2. 创建分支

```bash
git checkout -b feat/your-feature-name
```

!!! info "分支命名"

    - 新功能：`feat/<description>`
    - 修复：`fix/<description>`
    - 文档：`docs/<description>`
    - 测试：`test/<description>`

### 3. 编写代码

遵循下面的代码规范编写你的改动。

### 4. 运行测试

```bash
make lint      # 代码检查
make format    # 自动格式化
make test      # 运行完整测试套件
make smoke     # 冒烟测试
```

### 5. 提交 PR

将你的分支推送到 GitHub 并创建 Pull Request，描述改动内容和目的。

---

## 代码风格

| 工具 | 用途 |
|------|------|
| **black** | 代码格式化 |
| **isort** | 导入排序 |
| **ruff** | Lint 检查 |

```bash
# 一键格式化
make format

# 检查但不修改
make lint      # Ruff
make contract  # 课程与文档契约
```

---

## 新 Lesson 要求

添加新的课程 lesson 时，必须包含以下文件：

| 文件 | 要求 |
|------|------|
| `model.py` | 模型定义，纯 `nn.Module` |
| `data.py` | 数据加载，必须提供显式 fake 或内置 synthetic 离线路径 |
| `train.py` | 训练入口，使用 `dlhub/` 脚手架 |
| `README.md` | 课程文档，包含原理、架构、运行方式 |

!!! warning "离线测试必须通过"

    每个 lesson 必须能在无网络、纯 CPU 环境下运行：可选真实数据集的课程提供
    `--dataset fake`，其余课程直接使用内置数据。`make contract` 会检查入口和文档参数，
    新增训练链路还应补充针对性的 pytest。

---

## 代码约定

- **Python 版本**：3.10+
- **类型注解**：所有函数签名使用 type annotations
- **注释**：最少化注释，代码自文档化优先
- **文档字符串**：公开 API 使用 docstring

### 使用 dlhub/ 脚手架

```python
from dlhub.device import resolve_device
from dlhub.paths import build_run_paths
from dlhub.seed import set_seed

# 固定随机种子
set_seed(42)

# 自动设备选择
device_info = resolve_device("auto")

# 获取输出目录
paths = build_run_paths(track="vision", lesson="lesson_01", run_name="baseline")
```

---

## Make 命令速查

| 命令 | 功能 |
|------|------|
| `make lint` | 运行 Ruff 静态检查 |
| `make format` | 自动运行 isort、Black 和 Ruff 修复 |
| `make contract` | 检查课程入口、核心 CLI、文档命令和精选 Smoke 覆盖 |
| `make test` | 运行 pytest 完整测试套件 |
| `make smoke` | 运行覆盖 8 个 track 的精选离线冒烟测试 |
