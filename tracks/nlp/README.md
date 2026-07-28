# NLP 轨（自然语言处理）

目标：从最小的文本分类任务开始，掌握 NLP 的数据预处理、tokenizer/vocab、embedding、训练与评估闭环，并逐步走向 attention/transformer、NER、阅读理解等任务。

设计原则：

- **学习优先**：尽量少依赖大框架，让训练循环/数据管线“看得见”。
- **离线可跑**：优先提供 synthetic 数据集用于冒烟与理解，再扩展到真实数据集。

## Lessons

- `lesson_01_compact_text_classification/`：compact 文本分类（最小 tokenizer + embedding mean pooling）
- `lesson_02_compact_text_classification_transformer/`：compact 文本分类（Transformer encoder 最小实现）
- `lesson_03_compact_ner_bilstm/`：compact NER（BiLSTM 序列标注）
- `lesson_04_compact_seq2seq_attention_generation/`：compact 文本生成（Seq2Seq + Bahdanau Attention）
- `lesson_05_compact_text_classification_textcnn/`：compact 文本分类（TextCNN）
- `lesson_06_compact_text_classification_bilstm/`：compact 文本分类（BiLSTM）
- `lesson_07_reading_comprehension/`：compact 阅读理解（span prediction，预测答案起止位置）
- `lesson_08_compact_text_matching_biencoder/`：compact 文本匹配（双塔编码器 + 相似度检索）
- `lesson_09_compact_transformer_summarization/`：compact 摘要生成（Transformer encoder-decoder + teacher forcing）
- `lesson_10_compact_prompt_tuning_classifier/`：compact Prompt Tuning 文本分类（冻结 encoder + soft prompt）
- `lesson_11_compact_few_shot_text_classification/`：compact Few-shot 文本分类（episodic sampling + prototype 分类）
- `lesson_12_compact_in_context_text_classification/`：compact In-Context 文本分类（support examples + 无梯度提示推理）
- `lesson_13_compact_masked_language_modeling/`：compact Masked Language Modeling（masked token 预测 + 自监督预训练）
- `lesson_14_compact_contrastive_sentence_embedding/`：compact Contrastive Sentence Embedding（双视图增强 + 句向量对比学习）
- `lesson_15_compact_cross_encoder_reranking/`：compact Cross-Encoder Reranking（query-doc 拼接 + 成对排序）
- `lesson_16_compact_text_clustering/`：compact Text Clustering（句向量聚类 + 原型更新 + 无标签结构发现）
- `lesson_17_compact_text_anomaly_detection/`：compact Text Anomaly Detection（正常样本建模 + 距离阈值 + 异常得分）
- `lesson_18_compact_topic_modeling/`：compact Topic Modeling（主题混合 + BoW 重建 + 潜在主题发现）
- `lesson_19_compact_distilled_text_classifier/`：compact Distilled Text Classifier（teacher-student 蒸馏 + 轻量分类器）
- `lesson_20_compact_adversarial_text_classification/`：compact Adversarial Text Classification（对抗 token 替换 + 鲁棒分类 + 预测一致性）
- `lesson_21_compact_adversarial_example_detection/`：compact Adversarial Example Detection（检测短文本是否被对抗扰动）
- `lesson_22_compact_weak_supervision_text_classification/`：compact Weak-Supervision Text Classification（标注函数投票 + 软伪标签融合）
- `lesson_23_compact_sentence_denoising_autoencoder/`：compact Sentence Denoising Autoencoder（句子去噪重建 + 自监督序列恢复）
- `lesson_24_compact_meta_few_shot_text_classification/`：compact Meta Few-Shot Text Classification（episodic 元学习 + prototype 适配）
- `lesson_25_compact_low_shot_intent_detection/`：compact Low-Shot Intent Detection（少样本意图分类 + 轻量文本编码器）
- `lesson_26_compact_joint_intent_slot_parsing/`：compact Joint Intent + Slot Parsing（意图分类 + BIO 槽位联合预测）
- `lesson_27_compact_textual_entailment/`：compact Textual Entailment（前提-假设蕴含判别 + 双句编码分类）
- `lesson_28_compact_semantic_textual_similarity/`：compact Semantic Textual Similarity（双句相似度回归 + pooled embedding）
- `lesson_29_compact_dialog_state_tracking/`：compact Dialog State Tracking（多轮对话状态维护 + 多槽位联合分类）
- `lesson_30_compact_dialog_response_selection/`：compact Dialog Response Selection（上下文-候选响应匹配 + 响应排序）
- `lesson_31_compact_slot_carryover_prediction/`：compact Slot Carryover Prediction（历史槽位继承判别 + 多槽位二分类）
- `lesson_32_compact_dialog_act_prediction/`：compact Dialog Act Prediction（对话行为分类 + 轮次语气模式建模）
- `lesson_33_compact_dialog_intent_prediction/`：compact Dialog Intent Prediction（任务导向意图分类 + 餐厅/打车场景）
- `lesson_34_compact_dialog_policy_prediction/`：compact Dialog Policy Prediction（系统动作预测 + 对话策略分类）
- `lesson_35_compact_dialog_domain_prediction/`：compact Dialog Domain Prediction（餐厅/酒店/打车/天气域分类）
- `lesson_36_compact_dialog_slot_prediction/`：compact Dialog Slot Prediction（cuisine/area/party 多槽位分类）
- `lesson_37_compact_dialog_outcome_prediction/`：compact Dialog Outcome Prediction（resolved/pending/escalated 结果分类）
- `lesson_38_compact_dialog_satisfaction_prediction/`：compact Dialog Satisfaction Prediction（dissatisfied/neutral/satisfied 满意度分类）
- `lesson_39_compact_dialog_escalation_risk_prediction/`：compact Dialog Escalation Risk Prediction（low/medium/high 升级风险分类）
- `lesson_40_compact_dialog_priority_prediction/`：compact Dialog Priority Prediction（low/medium/high 优先级分类）
- `lesson_41_compact_dialog_transfer_prediction/`：compact Dialog Transfer Prediction（low/medium/high 转接需求分类）
- `lesson_42_compact_dialog_resolution_time_prediction/`：compact Dialog Resolution Time Prediction（short/medium/long 处理时长分类）
- `lesson_43_compact_dialog_callback_prediction/`：compact Dialog Callback Prediction（是否需要回拨的二分类）
- `lesson_44_compact_dialog_sla_breach_prediction/`：compact Dialog SLA Breach Prediction（是否 SLA breach 的二分类）
- `lesson_45_compact_dialog_followup_channel_prediction/`：compact Dialog Followup Channel Prediction（email/sms/call 三分类）
- `lesson_46_compact_dialog_reopen_prediction/`：compact Dialog Reopen Prediction（对话是否 reopen 的二分类）
- `lesson_47_compact_dialog_resolution_owner_prediction/`：compact Dialog Resolution Owner Prediction（billing/support/operations 三分类）
- `lesson_48_compact_dialog_resolution_action_prediction/`：compact Dialog Resolution Action Prediction（close/handoff/followup/resolve/escalate 五分类）
- `lesson_49_compact_dialog_owner_handoff_prediction/`：compact Dialog Owner Handoff Prediction（none/billing/support/operations 四分类）
