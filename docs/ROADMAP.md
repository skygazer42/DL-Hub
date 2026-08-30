# 学习路线图

DL-Hub 当前把课程组织为 **8 条学习赛道**。推荐路线不是硬性依赖：每个 lesson 都以可运行入口为边界，并提供离线数据路径和可重复的验收方式。

## 推荐顺序

1. **Foundations**：Tensor、autograd 与训练循环，建立公共基础。
2. **Vision**：从 LeNet 到视觉 Transformer，理解分类、检测与分割任务。
3. **NLP**：从文本分类到序列标注、注意力与 Transformer。
4. **GNN**：从图卷积到节点、边和图级任务。
5. **Point Cloud**：点集建模、邻域聚合、分类、分割与三维任务。
6. **Generative**：VAE、GAN、Diffusion 与 Flow Matching。
7. **LLM**：Causal LM、参数高效微调、指令学习与论文机制实验。
8. **Multimodal**：视觉语言对齐、检索、问答和跨模态融合。

各赛道的当前 lesson 清单和准确命令见[学习赛道总览](tracks/index.md)。统计数字由 `python scripts/project_stats.py --write` 维护，不在本页复制易漂移的快照。

## 按时间选择路线

| 路线 | 建议内容 | 适合目标 |
|---|---|---|
| Weekend Sprint | Foundations 2 课 → Vision 前 2 课 → Generative 第 1 课 → LLM 第 1 课 | 快速建立完整训练闭环 |
| Two-Week Deep Dive | Foundations → Vision 5 课 → NLP 4 课 → GNN 3 课 → Generative 2 课 → LLM 1 课 → Point Cloud 1 课 | 横向理解主要任务范式 |
| Full Curriculum | 按上述推荐顺序完成全部 8 条赛道 | 系统学习并进行机制对照 |

## 每个 lesson 如何学习

1. 先读 lesson 的 `README.md`，确认目标、输入、指标与验收标准。
2. 用 `python scripts/run_lesson.py <track> <lesson> --describe` 查询真实 CLI。
3. 先运行 fake 或内置 synthetic 离线路径，再切换真实数据或扩大训练规模。
4. 检查 `outputs/<track>/<lesson>/<run_name>/` 下的配置、指标与 checkpoint。
5. 修改一个机制并重复实验，而不是只记录一次“能运行”。

训练型 lesson 通常包含 `model.py`、`data.py`、`train.py` 和 `README.md`；基础演示或特殊任务可以精简，但必须满足[实现契约](implementation-contract.md)和[运行约定](CONVENTIONS.md)。

## 验证强度

| 层级 | 命令 | 能证明什么 |
|---|---|---|
| 静态契约 | `make verify` | 代码风格、课程入口、命名叙事与 Zoo 审计元数据一致 |
| 精选运行 | `make smoke` | 8 条赛道的代表性离线链路可运行 |
| 针对性测试 | `python -m pytest -q tests/<file>.py` | 本次改动涉及的行为没有回归 |
| 完整测试 | `make test` | 仓库测试套件在当前环境通过 |

这些层级不能互相替代；需要依据改动范围选择足以支撑结论的验证。开发者完整流程见[测试指南](developer/testing.md)。
