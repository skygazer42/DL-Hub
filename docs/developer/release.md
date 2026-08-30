# 发布检查

本页描述可安装 `dlhub` 包的工程边界和发布前验证。CI **只构建并保存产物，不自动发布到 PyPI**；真正发布仍需维护者显式决定版本与目标仓库。

## 事实源

| 内容 | 唯一维护位置 |
|---|---|
| 包版本 | `dlhub/__about__.py`（`pyproject.toml` 动态读取） |
| 运行时与可选依赖 | `pyproject.toml` 的 `dependencies` / `optional-dependencies` |
| 兼容安装命令 | `requirements*.txt`，只转发到对应 extra |
| wheel 中可导入包范围 | `pyproject.toml` 的 `include = ["dlhub*"]` |
| 开发者命令 | `Makefile` |

`tracks/`、`Llms/`、`scripts/` 和课程资料不会进入 wheel；它们应从 Git 仓库根目录运行。

## 本地发布前检查

先按[安装指南](../getting-started/installation.md)安装适配平台的 PyTorch，再安装完整开发工具：

```bash
python -m pip install -e ".[all,dev,docs]"
python -m pip check
make release-check
```

`make release-check` 组合执行：

1. `make verify`：lint、lesson contract、生成统计、Zoo 完整性、叙事和保真度元数据；
2. `make test`：完整 pytest；
3. `make docs`：`mkdocs build --strict`；
4. `make package`：先清除 `dist/` 中仅匹配 `dlhub-*.whl` / `dlhub-*.tar.gz` 的陈旧产物，再构建唯一一组 sdist/wheel，审计归档内容并运行 `twine check`；
5. `make package-smoke`：分别验证刚构建的 wheel，以及从 sdist 在临时环境中重建的 wheel。

内容审计会拒绝绝对路径、`..`、Windows 路径、重复成员、链接/特殊文件、私钥材料、凭据类文件名和非预期顶层目录；同时核对体积上限、成员数、SHA-256、README、LICENSE、`pyproject.toml`、版本、Python 下限与基础依赖元数据。wheel 只能包含 `dlhub` 和对应的 `.dist-info`，不能包含 `tracks`、`scripts`、`resources` 或测试。

setuptools 当前默认把根目录 `tests/` 源文件列入 sdist，但它们不会安装到 `site-packages`。其中许多仓库级测试依赖不随发行包提供的课程目录，因此不要把这批文件当作可脱离 Git 仓库运行的完整项目测试；wheel 仍严格排除所有测试文件。

隔离安装门禁会在不安装 PyTorch 的环境调用 NumPy 指标 API，核对 `dlhub.__version__` 与 `importlib.metadata`，并确认仓库专用的 `tracks` 不可导入。它还会在仓库外的临时 venv 从 sdist 重建 wheel，再在另一个临时 venv 安装和复核相同边界。实现只使用 Python 标准库管理 venv、解释器路径和临时目录，不依赖平台特定的 shell 临时目录命令；pip 缓存也放在自动清理的临时目录中。

该门禁会让临时 venv 的 pip 从当前配置的软件源解析 NumPy，因此需要网络或已配置的内部镜像；wheel 本身始终从本地 `dist/` 安装。

训练链路、依赖或设备逻辑有变化时，还应单独执行 `make smoke`；它不会被静态检查或单元测试替代。

## CI 证据

- `python-ci.yml / tests (Python 3.10)`：最低支持版本执行仓库门禁、打包、wheel 隔离安装、sdist 隔离重建/安装和完整测试，并保存 7 天构建产物。
- `python-ci.yml / tests (Python 3.12)`：同一矩阵中的兼容版本执行完整测试。
- `docs.yml / build`：文档相关 PR 执行 strict 构建；合并到 `main` 后，只有仓库启用 GitHub Pages 才部署。

发布前应检查这些状态均成功，并确认生成产物中的版本、README、许可证、依赖和项目链接符合预期。
