# 代码风格与目录规范

目标：让整个仓库像“一个项目”而不是“拼盘”。

## 语言与注释

- 课程说明与 README：中文优先，必要时保留英文术语（例如 *overfitting*）。
- 代码注释：少而精，优先写在 `README.md` 与函数 docstring。

## Python 版本与类型标注

- Python 3.10+
- 类型标注：能标就标，保持可读（`X | None` 优先）。

## 代码格式化与静态检查

- 格式化：`black`
- import：`isort`
- lint：`ruff`

常用命令（仓库根目录）：

```bash
make lint
make format
make verify
# 只对本次改动选择对应的 pytest 文件或测试函数
```

## 目录命名约定

- 统一用小写 + 下划线：`lesson_01_mnist_lenet`
- 避免空格与大小写混用（历史目录会逐步被新结构替代）
- 缩放课程/模型使用 `compact`，程序生成数据使用 `synthetic`，共享实现工厂使用 `baseline`

三类命名的边界与保真度升级规则见
[实现契约：从课程到可验证系统](implementation-contract.md)。

## 训练脚手架约定（高一致性）

新实现尽量复用 `dlhub/`：

- `dlhub.seed`：随机种子
- `dlhub.device`：设备选择
- `dlhub.training.loop`：训练/评估循环
- `dlhub.paths`：输出目录结构

不要在每个 lesson 自己“再发明一次”训练循环。

## 输出目录

统一输出到仓库根目录的 `outputs/` 下（见 `docs/CONVENTIONS.md`）。

## 依赖管理

- `requirements.txt`：最小运行依赖
- `requirements-dev.txt`：开发工具
- `requirements-vision.txt` / `requirements-nlp.txt` 等：按轨道可选依赖
