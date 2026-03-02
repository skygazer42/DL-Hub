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
