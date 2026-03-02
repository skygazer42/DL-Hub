# LLM 轨（大模型相关的最小可跑实验）

目标：用**可控的 toy 任务**把语言模型里最关键的结构与训练闭环跑通（tokenization → 数据 → 模型 → loss → 生成 → 记录产物）。

原则：

- 依赖尽量少（优先纯 PyTorch）
- 先可跑通，再逐步扩展规模与技巧
- 所有 lesson 统一输出到 `outputs/llm/<lesson>/<run_name>/`

## Lessons

- `lesson_01_toy_causal_lm_transformer/`：toy causal LM（Transformer decoder + 自回归生成）

