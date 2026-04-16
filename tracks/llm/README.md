# LLM 轨（大模型相关的最小可跑实验）

目标：用**可控的 toy 任务**把语言模型里最关键的结构与训练闭环跑通（tokenization → 数据 → 模型 → loss → 生成 → 记录产物）。

原则：

- 依赖尽量少（优先纯 PyTorch）
- 先可跑通，再逐步扩展规模与技巧
- 所有 lesson 统一输出到 `outputs/llm/<lesson>/<run_name>/`

## Lessons

- `lesson_01_toy_causal_lm_transformer/`：toy causal LM（Transformer decoder + 自回归生成）
- `lesson_02_toy_chat_sft/`：toy chat SFT（chat-format prompt + assistant-only masked loss）
- `lesson_03_toy_mamba_language_model/`：toy Mamba 语言模型（状态空间混合 + 自回归预测）
- `lesson_04_toy_instruction_tuning/`：toy 指令微调（single-turn prompt + response-only masked loss）
- `lesson_05_toy_prefix_tuning/`：toy Prefix Tuning（冻结 decoder LM + 可训练前缀向量）
- `lesson_06_toy_preference_optimization/`：toy 偏好优化（chosen/rejected 对比 + DPO 风格目标）
- `lesson_07_toy_reward_modeling/`：toy 奖励建模（成对排序 + 标量奖励头）
- `lesson_08_toy_span_corruption/`：toy Span Corruption（连续片段掩码 + 去噪解码）
- `lesson_09_toy_rlhf_ppo/`：toy RLHF PPO（token 级奖励 + clipped PPO 更新）
- `lesson_10_toy_grpo_alignment/`：toy GRPO Alignment（grouped rollouts + 相对优势归一化 + 响应级偏好优化）
- `lesson_11_toy_rag_language_model/`：toy RAG 语言模型（检索 doc_id + 文档条件解码 + 检索增强生成）
- `lesson_12_toy_transformer_interpretability/`：toy Transformer 可解释性（attention map 检查 + token saliency + 局部解释）
- `lesson_13_toy_tool_calling_agent/`：toy Tool-Calling Agent（工具选择 + 参数生成 + 小型代理循环）
- `lesson_14_toy_replaced_token_detection_transformer/`：toy Replaced-Token Detection Transformer（替换 token 判别 + 编码式自监督预训练）
- `lesson_15_toy_llm_judge/`：toy LLM Judge（prompt-answer 打分 + 候选质量排序）
- `lesson_16_toy_multi_turn_memory_sft/`：toy Multi-Turn Memory Chat SFT（多轮历史拼接 + assistant-only masked loss）
- `lesson_17_toy_self_refine_prompting/`：toy Self-Refine Prompting（草稿-批评-修订链路 + 提示式自改写监督）
- `lesson_18_toy_reflection_memory_agent/`：toy Reflection Memory Agent（反思写入记忆 + 检索式答案修订）
- `lesson_19_toy_plan_execute_prompting/`：toy Plan-Execute Prompting（计划提示 + 执行提示两阶段监督）
- `lesson_20_toy_react_tool_prompting/`：toy ReAct Tool Prompting（思考-行动交替轨迹 + 工具选择监督）
- `lesson_21_toy_tree_of_thought_prompting/`：toy Tree-of-Thought Prompting（多分支候选 + 路径选择 + 最终答案监督）
- `lesson_22_toy_self_consistency_prompting/`：toy Self-Consistency Prompting（多样答案采样 + 投票一致性 + 最终答案监督）
- `lesson_23_toy_critic_rerank_prompting/`：toy Critic-Rerank Prompting（候选回答打分 + critique 上下文 + 最优答案重排）
- `lesson_24_toy_debate_prompting/`：toy Debate Prompting（正反论点提示 + judge 标记 + verdict 监督）
- `lesson_25_toy_verifier_guided_prompting/`：toy Verifier-Guided Prompting（草稿-验证-修正链路 + guide token 监督）
- `lesson_26_toy_process_supervision_prompting/`：toy Process Supervision Prompting（草稿-检查-流程监督链路 + process token 监督）
- `lesson_27_toy_self_correction_prompting/`：toy Self-Correction Prompting（草稿-批评-自修正链路 + corrected span 监督）
- `lesson_28_toy_reference_grounded_prompting/`：toy Reference-Grounded Prompting（显式 reference span + grounded token 监督）
- `lesson_29_toy_constraint_repair_prompting/`：toy Constraint-Repair Prompting（约束检查-修复链路 + repair token 监督）
- `lesson_30_toy_citation_grounded_prompting/`：toy Citation-Grounded Prompting（引用 span 拷贝监督 + cite token 约束）
- `lesson_31_toy_schema_constrained_prompting/`：toy Schema-Constrained Prompting（schema marker 监督 + 结构化字段续写）
- `lesson_32_toy_json_constrained_prompting/`：toy JSON-Constrained Prompting（json marker 监督 + JSON 字段续写）
- `lesson_33_toy_function_signature_prompting/`：toy Function-Signature Prompting（call marker 监督 + 函数签名续写）
- `lesson_34_toy_xml_constrained_prompting/`：toy XML-Constrained Prompting（xml marker 监督 + XML 结构续写）
- `lesson_35_toy_regex_constrained_prompting/`：toy Regex-Constrained Prompting（regex marker 监督 + 模式约束字段续写）
- `lesson_36_toy_ebnf_constrained_prompting/`：toy EBNF-Constrained Prompting（ebnf marker 监督 + 规则约束续写）
- `lesson_37_toy_sql_constrained_prompting/`：toy SQL-Constrained Prompting（sql marker 监督 + 查询骨架续写）
- `lesson_38_toy_yaml_constrained_prompting/`：toy YAML-Constrained Prompting（yaml marker 监督 + key-value 行续写）
- `lesson_39_toy_csv_constrained_prompting/`：toy CSV-Constrained Prompting（csv marker 监督 + 表头/行续写）
- `lesson_40_toy_toml_constrained_prompting/`：toy TOML-Constrained Prompting（toml marker 监督 + key=value 续写）
- `lesson_41_toy_markdown_table_constrained_prompting/`：toy Markdown-Table Constrained Prompting（table marker 监督 + 表头/表格行续写）
- `lesson_42_toy_ini_constrained_prompting/`：toy INI-Constrained Prompting（ini marker 监督 + section/key=value 续写）
- `lesson_43_toy_tsv_constrained_prompting/`：toy TSV-Constrained Prompting（tsv marker 监督 + column/value 行续写）
