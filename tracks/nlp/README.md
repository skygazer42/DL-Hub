# NLP 轨（自然语言处理）

目标：从最小的文本分类任务开始，掌握 NLP 的数据预处理、tokenizer/vocab、embedding、训练与评估闭环，并逐步走向 attention/transformer、NER、阅读理解等任务。

设计原则：

- **学习优先**：尽量少依赖大框架，让训练循环/数据管线“看得见”。
- **离线可跑**：优先提供 toy/synthetic 数据集用于冒烟与理解，再扩展到真实数据集。

## Lessons

- `lesson_01_toy_text_classification/`：toy 文本分类（最小 tokenizer + embedding mean pooling）
- `lesson_02_toy_text_classification_transformer/`：toy 文本分类（Transformer encoder 最小实现）
- `lesson_03_toy_ner_bilstm/`：toy NER（BiLSTM 序列标注）
- `lesson_04_toy_seq2seq_attention_generation/`：toy 文本生成（Seq2Seq + Bahdanau Attention）
- `lesson_05_toy_text_classification_textcnn/`：toy 文本分类（TextCNN）
- `lesson_06_toy_text_classification_bilstm/`：toy 文本分类（BiLSTM）
- `lesson_07_reading_comprehension/`：toy 阅读理解（span prediction，预测答案起止位置）
- `lesson_08_toy_text_matching_biencoder/`：toy 文本匹配（双塔编码器 + 相似度检索）
- `lesson_09_toy_transformer_summarization/`：toy 摘要生成（Transformer encoder-decoder + teacher forcing）
- `lesson_10_toy_prompt_tuning_classifier/`：toy Prompt Tuning 文本分类（冻结 encoder + soft prompt）
- `lesson_11_toy_few_shot_text_classification/`：toy Few-shot 文本分类（episodic sampling + prototype 分类）
- `lesson_12_toy_in_context_text_classification/`：toy In-Context 文本分类（support examples + 无梯度提示推理）
- `lesson_13_toy_masked_language_modeling/`：toy Masked Language Modeling（masked token 预测 + 自监督预训练）
- `lesson_14_toy_contrastive_sentence_embedding/`：toy Contrastive Sentence Embedding（双视图增强 + 句向量对比学习）
- `lesson_15_toy_cross_encoder_reranking/`：toy Cross-Encoder Reranking（query-doc 拼接 + 成对排序）
- `lesson_16_toy_text_clustering/`：toy Text Clustering（句向量聚类 + 原型更新 + 无标签结构发现）
- `lesson_17_toy_text_anomaly_detection/`：toy Text Anomaly Detection（正常样本建模 + 距离阈值 + 异常得分）
- `lesson_18_toy_topic_modeling/`：toy Topic Modeling（主题混合 + BoW 重建 + 潜在主题发现）
- `lesson_19_toy_distilled_text_classifier/`：toy Distilled Text Classifier（teacher-student 蒸馏 + 轻量分类器）
- `lesson_20_toy_adversarial_text_classification/`：toy Adversarial Text Classification（对抗 token 替换 + 鲁棒分类 + 预测一致性）
- `lesson_21_toy_adversarial_example_detection/`：toy Adversarial Example Detection（检测短文本是否被对抗扰动）
- `lesson_22_toy_weak_supervision_text_classification/`：toy Weak-Supervision Text Classification（标注函数投票 + 软伪标签融合）
- `lesson_23_toy_sentence_denoising_autoencoder/`：toy Sentence Denoising Autoencoder（句子去噪重建 + 自监督序列恢复）
- `lesson_24_toy_meta_few_shot_text_classification/`：toy Meta Few-Shot Text Classification（episodic 元学习 + prototype 适配）
- `lesson_25_toy_low_shot_intent_detection/`：toy Low-Shot Intent Detection（少样本意图分类 + 轻量文本编码器）
- `lesson_26_toy_joint_intent_slot_parsing/`：toy Joint Intent + Slot Parsing（意图分类 + BIO 槽位联合预测）
- `lesson_27_toy_textual_entailment/`：toy Textual Entailment（前提-假设蕴含判别 + 双句编码分类）
- `lesson_28_toy_semantic_textual_similarity/`：toy Semantic Textual Similarity（双句相似度回归 + pooled embedding）
- `lesson_29_toy_dialog_state_tracking/`：toy Dialog State Tracking（多轮对话状态维护 + 多槽位联合分类）
- `lesson_30_toy_dialog_response_selection/`：toy Dialog Response Selection（上下文-候选响应匹配 + 响应排序）
- `lesson_31_toy_slot_carryover_prediction/`：toy Slot Carryover Prediction（历史槽位继承判别 + 多槽位二分类）
- `lesson_32_toy_dialog_act_prediction/`：toy Dialog Act Prediction（对话行为分类 + 轮次语气模式建模）
- `lesson_33_toy_dialog_intent_prediction/`：toy Dialog Intent Prediction（任务导向意图分类 + 餐厅/打车场景）
- `lesson_34_toy_dialog_policy_prediction/`：toy Dialog Policy Prediction（系统动作预测 + 对话策略分类）
- `lesson_35_toy_dialog_domain_prediction/`：toy Dialog Domain Prediction（餐厅/酒店/打车/天气域分类）
- `lesson_36_toy_dialog_slot_prediction/`：toy Dialog Slot Prediction（cuisine/area/party 多槽位分类）
- `lesson_37_toy_dialog_outcome_prediction/`：toy Dialog Outcome Prediction（resolved/pending/escalated 结果分类）
- `lesson_38_toy_dialog_satisfaction_prediction/`：toy Dialog Satisfaction Prediction（dissatisfied/neutral/satisfied 满意度分类）
- `lesson_39_toy_dialog_escalation_risk_prediction/`：toy Dialog Escalation Risk Prediction（low/medium/high 升级风险分类）
- `lesson_40_toy_dialog_priority_prediction/`：toy Dialog Priority Prediction（low/medium/high 优先级分类）
- `lesson_41_toy_dialog_transfer_prediction/`：toy Dialog Transfer Prediction（low/medium/high 转接需求分类）
- `lesson_42_toy_dialog_resolution_time_prediction/`：toy Dialog Resolution Time Prediction（short/medium/long 处理时长分类）
- `lesson_43_toy_dialog_callback_prediction/`：toy Dialog Callback Prediction（是否需要回拨的二分类）
- `lesson_44_toy_dialog_sla_breach_prediction/`：toy Dialog SLA Breach Prediction（是否 SLA breach 的二分类）
- `lesson_45_toy_dialog_followup_channel_prediction/`：toy Dialog Followup Channel Prediction（email/sms/call 三分类）
- `lesson_46_toy_dialog_reopen_prediction/`：toy Dialog Reopen Prediction（对话是否 reopen 的二分类）
- `lesson_47_toy_dialog_resolution_owner_prediction/`：toy Dialog Resolution Owner Prediction（billing/support/operations 三分类）
- `lesson_48_toy_dialog_resolution_action_prediction/`：toy Dialog Resolution Action Prediction（close/handoff/followup/resolve/escalate 五分类）
- `lesson_49_toy_dialog_owner_handoff_prediction/`：toy Dialog Owner Handoff Prediction（none/billing/support/operations 四分类）
