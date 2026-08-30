# 测试指南

DL-Hub 使用静态契约、精选 smoke、针对性 pytest 和完整 CI 四层验证。
默认从改动能证明的最小范围开始；测试数量不是目标，结论与改动范围匹配才是目标。

pytest 配置只维护在根目录 `pyproject.toml` 的 `[tool.pytest.ini_options]`；默认启用未知配置与未知 marker 严格检查，不再维护第二份 `pytest.ini`。

完整分层定义见 [实现契约](../implementation-contract.md)。

---

## 运行测试

### 日常改动：先做快速仓库验证

```bash
# lint + lesson contract + 生成统计 + Zoo 完整性/保真度 + 命名叙事
make verify

# 单独确认 README / docs 的生成统计块没有漂移（make verify 已包含）
make stats

# 单独复核全部本地 Zoo 注册表（make verify 已包含）
python scripts/zoo_integrity.py --check

# 复核 339 门数据/benchmark 分类、真实数据 profile 和历史运行证明
make evidence

# 更强的手工检查：额外构建六个领域代表并运行前向/模拟
python scripts/zoo_integrity.py --check --smoke

# 全量运行 339 个 lesson 的 python -m ... --help（通常约 4–7 分钟，不属于 make verify）
make lesson-entrypoints
```

这些检查不训练模型。随后只运行与修改文件直接相关的测试：

`make lesson-entrypoints` 使用 CPU、离线环境变量和隔离的临时 cwd/缓存/输出目录；
它不做 socket 级网络阻断，因此证明的是入口导入与 CLI 帮助路径，不是完整训练或绝对断网。

```bash
# 单个测试文件
pytest -q tests/test_kmeans.py

# 单个关键行为
pytest -q tests/test_kmeans.py::test_kmeans_clusters_separable_points
```

### 何时扩大范围

- 改动训练链路：运行对应 lesson 的 smoke 或短训练测试。
- 改动共享底层模块：运行该模块测试和受影响的代表性 track 测试。
- 改动注册表/发现逻辑：运行对应 Zoo、统计或 lesson contract 测试。
- 改动文档或 MkDocs 配置：运行 `make docs`，确保 strict 构建通过。
- 改动依赖、打包或发布元数据：运行 `make package-smoke`。
- 发布前或 CI：运行 `pytest -q` / `make test` 完整套件。

### 查看测试覆盖率

```bash
pytest --cov=dlhub --cov=ml_algorithms --cov-report=term-missing
```

---

## 冒烟测试

精选冒烟测试从 8 个 track 各取代表性课程，验证离线 CPU 训练链路和核心依赖。
它不是 339 个课程的全量训练；全量入口与文档参数由静态课程契约检查覆盖。

```bash
# 运行全局冒烟测试
python scripts/smoke_check.py

# 或通过 Make
make smoke

# 查看精选覆盖清单
python scripts/smoke_check.py --list

# 检查全部 lesson 的静态契约
make contract
```

!!! tip "冒烟测试的意义"

    冒烟测试确保：

    - 8 个学习 track 都至少有一个代表性案例
    - 代表性模型的前向传播和短训练链路正常
    - 输出配置、指标和检查点能够落盘
    - 不依赖外部数据下载或 GPU

---

## 编写测试

### 为新 Lesson 编写测试

不要因为新增一个目录就机械复制三套测试。先判断现有 contract 和 curated smoke 是否已经覆盖；
只有存在新的行为边界、计算机制或回归风险时才增加针对性测试。例如：

```python
# tests/test_vision_lesson_XX.py
import torch


def test_conditioning_changes_prediction():
    """验证本课新增的条件机制真实参与计算。"""
    from tracks.vision.lesson_XX.model import MyModel

    model = MyModel()
    x = torch.randn(2, 3, 32, 32)
    condition_a = torch.zeros(2, 8)
    condition_b = torch.ones(2, 8)
    assert not torch.allclose(model(x, condition_a), model(x, condition_b))
```

形状、CLI 和离线入口如果已由共享测试或静态 contract 保证，不在每个 lesson 重复断言。

### 测试文件命名

```text
tests/
├── test_linear_models.py       # ML 算法测试
├── test_kmeans.py              # ML 算法测试
├── test_optimizers.py          # 优化器测试
├── test_vision_lesson_01.py    # 课程冒烟测试
├── test_detection_zoo.py       # Zoo CLI 测试
└── ...
```

!!! note "命名约定"

    - ML 算法测试：`test_<algorithm_name>.py`
    - 课程测试：`test_<track>_lesson_<XX>.py`
    - Zoo 测试：`test_<zoo_name>.py`
    - 工具测试：`test_<tool_name>.py`

---

## CI 集成

DL-Hub 使用 **GitHub Actions** 进行持续集成。

### CI 流水线

```mermaid
graph LR
    A[Push / PR] --> B[Python 3.10 仓库门禁]
    B --> C[sdist / wheel + 隔离安装校验]
    C --> D[Python 3.10 完整测试]
    A --> E[Python 3.12 完整测试]
    A --> F[文档改动: MkDocs strict]
```

### CI 检查项

| 检查 | 命令 | 说明 |
|------|------|------|
| Lint | `make lint` | Ruff 静态检查 |
| Lesson Contracts | `make contract` | 全量入口、核心 CLI、文档命令与精选 Smoke 覆盖 |
| Generated Stats | `make stats` | 检查 README 与文档统计块是否匹配当前仓库 |
| Zoo Integrity | `make zoo-integrity` | 全量注册表离线导入、ID、排序与 builder 映射 |
| Narrative | `make narrative` | 命名边界与 lesson 路径一致性 |
| Fidelity | `make fidelity` | Model Zoo 审计元数据与源码证据 |
| Evidence | `make evidence` | 339 门数据/benchmark 分类、真实数据 profile 与历史运行证明 |
| Tests | `pytest -q` | 全量单元测试 |
| Package | `make package` | 构建 sdist/wheel 并用 Twine 校验元数据 |
| Package Smoke | `make package-smoke` | 隔离安装 wheel，并从 sdist 隔离重建、安装和验证同一发行边界 |
| Docs | `make docs` | 严格构建 MkDocs；文档 PR 合并前执行 |

`python-ci.yml` 的单一矩阵定义覆盖最低支持版本 3.10 和兼容版本 3.12；只有最低版本执行
仓库、统计和打包门禁，两个版本都运行完整测试。`make smoke` 是本地或发布前的代表性运行
验证，目前不在常规 GitHub Actions 中重复执行。

### Actions 供应链与权限边界

- 所有外部 `uses:` 都固定到官方 release 对应的完整 commit SHA；同行版本注释供审阅，也让 Dependabot 在更新 SHA 时同步维护可读版本。
- `.github/dependabot.yml` 的 `github-actions` 生态会按月检查 Action 更新；不要把固定 SHA 手工退回可移动的 major tag。
- 普通 CI 和文档构建只授予 `contents: read`，checkout 不把 token 持久化到本地 Git 配置。
- PR（包括 fork PR）只构建和测试，不上传或部署 Pages；Pages 写权限与 OIDC 只授予 `deploy` job。
- 手动触发可验证任意分支的文档，但只有 `main` 且仓库已启用 Pages 时才会发布；同一 ref 的旧构建会取消，进行中的生产部署不会被新运行中断。
- Python job 最长运行 60 分钟，文档构建和部署分别限制为 20 与 10 分钟。

!!! warning "PR 合并前提"

    所有 CI 检查必须通过后 PR 才能合并。
    如果测试失败，请查看 Actions 日志定位问题。
