# DL-Hub PyTorch 统一课程化重置（100 Tasks）Implementation Plan

**Goal:** 把这个仓库重置成一个“能系统学到东西”的 PyTorch 学习项目：统一代码风格、统一训练/评估脚手架、统一目录结构，并将现有零散项目按多轨课程重写成可运行/可练习/可验收的学习路径；PDF 资料完整保留。

**Architecture:** “一个引擎 + 多条轨道”。引擎放在 `dlhub/`（数据、训练、评估、日志、复现实验）；课程放在 `tracks/`（foundations/vision/nlp/gnn/pointcloud/generative/llm），每节课都使用同一套脚手架与约定。旧代码作为参考源，在新实现通过验收后再删除对应旧目录/旧脚本（保留在 Git 历史里，可追溯）。

**Tech Stack:** Python 3.10+, PyTorch 2.x, NumPy, torchvision（vision 轨），pytest, ruff, black, isort（已有基线）。

**Important constraints:**
- 不做 git commit（按用户要求）。
- PDF 文件不删除、不改内容；允许后续做索引/标签化。
- “先实现后删除”：每个旧目录在删除前必须有等价（或简化实现）新实现 + 运行说明 + 最小验收脚本/测试。
- 删除动作不逐次询问：所有删除决策与验收证据记录在 `docs/DELETE_LIST.md`，保证可追溯。

---

## 里程碑总览

- M0：统一工程基线（结构/规范/工具/脚手架骨架）
- M1：`dlhub` 训练引擎最小闭环（CPU 可跑、可复现）
- M2：Foundations 轨（PyTorch 基础 + 对照 NumPy 原理）
- M3：Vision 轨（从 MNIST 到更现代训练范式）
- M4：NLP 轨（把 TF/Keras 旧内容用 PyTorch 统一重写）
- M5：GNN 轨（用纯 PyTorch 重写 GCN/GAT/GIN 最小实现）
- M6：PointCloud 轨（PointNet/DGCNN 最小实现）
- M7：Generative 轨（VAE/GAN 最小实现）
- M8：LLM 轨（围绕 PDF 资料做“可跑的小实验 + 阅读路线”）
- M9：删减收敛（移除旧实现，保证整体一致）

---

## Task List（100）

### M0：统一工程基线（1–12）

1. 新增 `docs/ROADMAP.md`：给出学习路线图（多轨并行但同一引擎）。
2. 新增 `docs/STYLEGUIDE.md`：统一代码风格/目录约定/命名/注释语言。
3. 新增 `docs/CONVENTIONS.md`：实验记录、seed、device、日志、checkpoint 规范。
4. 将根 `README.md` 重写为“课程入口”（明确主线、先修、如何开始、如何验收）。
5. 新增 `docs/FAQ.md`：常见坑（CUDA/CPU、数据下载、路径、依赖）。
6. 新增 `docs/INSTALL.md`：最小依赖 + 可选依赖（vision/gnn/nlp）。
7. 新增 `requirements.txt`（runtime 最小集合）与 `requirements-*.txt`（可选轨道依赖）。
8. 更新 `Makefile`：新增 `make smoke`、`make lint` 覆盖 `dlhub/`、`tracks/`。
9. 新增 `scripts/doctor.py`：检查 Python/torch/torchvision 版本、CUDA 可用性与常见冲突。
10. 新增 `scripts/new_lesson.py`：生成 lesson 模板（README + train/eval/model/data）。
11. 增强 `scripts/smoke_check.py`：加入对 `dlhub` 与至少 1 个 lesson 的冒烟。
12. 调整 `.gitignore`：不要全局忽略所有图片；改为只忽略 `outputs/`、`runs/` 产物目录。

### M1：训练引擎 `dlhub`（13–34）

13. 创建包 `dlhub/__init__.py` 与版本号 `dlhub/__about__.py`。
14. `dlhub/seed.py`：统一随机种子（python/numpy/torch + cuda）。
15. `dlhub/device.py`：设备选择策略（cpu/cuda/mps）+ 显示信息。
16. `dlhub/config.py`：dataclass 配置（训练/数据/模型/日志）+ CLI 合并策略。
17. `dlhub/logging.py`：统一 logger（stdout + 可选写文件）。
18. `dlhub/paths.py`：统一 `outputs/` 目录结构（按 track/lesson/exp_name）。
19. `dlhub/metrics.py`：通用分类 accuracy、topk、平均器。
20. `dlhub/progress.py`：轻量进度条封装（可选 tqdm）。
21. `dlhub/checkpoint.py`：保存/加载 state_dict（模型/优化器/epoch/配置摘要）。
22. `dlhub/training/loop.py`：通用 fit/evaluate 训练循环（支持 hooks）。
23. `dlhub/training/hooks.py`：hooks 接口（on_step/on_epoch_end）。
24. `dlhub/training/early_stop.py`：早停（简化实现）。
25. `dlhub/training/ema.py`：EMA（可选，高阶课）。
26. `dlhub/data/splits.py`：train/val split helper（可复现）。
27. `dlhub/data/compact.py`：compact 数据集（用于引擎单测，不依赖下载）。
28. `dlhub/nn/modules.py`：常用模块（MLP/ConvBlock）可读实现。
29. `dlhub/nn/init.py`：权重初始化工具（学习用）。
30. `dlhub/eval/confusion.py`：混淆矩阵（小工具）。
31. `dlhub/cli.py`：统一入口（可选）：`python -m dlhub ...`
32. `tests/test_dlhub_seed.py`：seed 可复现性测试（compact 数据）。
33. `tests/test_dlhub_loop.py`：训练循环在 compact 分类上 loss 下降/acc 上升。
34. `scripts/benchmark_cpu.py`：CPU 训练速度粗测（防止写出巨慢循环）。

### M2：Foundations 轨（35–46）

35. 创建 `tracks/foundations/README.md`：这条轨道学什么。
36. Lesson 01：tensor/shape/广播（含练习与验收）。
37. Lesson 02：autograd 与反向传播（含小实验：手写线性回归对照 autograd）。
38. Lesson 03：优化器与学习率（对照 `optimization/python`）。
39. Lesson 04：过拟合与正则（L2/Dropout/数据增强的理论入口）。
40. Lesson 05：训练稳定性（梯度裁剪、初始化、归一化）。
41. 把 `ml_algorithms/python` 内容映射到 “原理补充” 页（不再作为主入口）。
42. 为 `ml_algorithms/python` 增加一个 “对照 torch” 的小示例脚本（同数据同指标）。
43. 新增 `tracks/foundations/exercises/`：用 pytest 作为练习验收（TODO 测试）。
44. 新增 `tests/test_foundations_lesson_smoke.py`：至少 1 节课可跑通。
45. 新增 `docs/LEARNING_RULES.md`：如何做练习、如何提交（即便不 commit，也要能自检）。
46. 把 foundations 轨加入 `scripts/smoke_check.py` 验证路径。

### M3：Vision 轨（47–64）

47. 创建 `tracks/vision/README.md`：vision 学习路径与数据依赖说明。
48. Vision Lesson 01：MNIST + LeNet（从 `Deep_project/Mnist/LeNet` 重写成统一结构）。
49. Vision Lesson 02：MNIST + MLP（对照 `Deep_project/Mnist/mlp`，统一脚手架）。
50. Vision Lesson 03：CIFAR10 + 小型 CNN（简化实现）。
51. Vision Lesson 04：训练技巧（augmentation、label smoothing、mixup 可选）。
52. Vision Lesson 05：迁移学习（torchvision resnet18，冻结/解冻）。
53. Vision Lesson 06：可解释性（Grad-CAM 简化实现）。
54. Vision Lesson 07：简单检测（用 torchvision detection 做简化实现，替代大 YOLOv5 工程）。
55. Vision Lesson 08：简单分割（小 U-Net，synthetic 或小数据）。
56. 把 `Deep_project/swin` 内容提炼成 “阅读与实验” 页，不直接搬原工程。
57. 把 `FCOS_Pytorch_case`/`retiannet` 做成 “进阶链接与差异解释” 页。
58. 为 vision 轨新增 `tracks/vision/tests/`：至少 MNIST lesson 有快速单测。
59. 新增 `scripts/download_datasets.py`：集中管理数据下载（可选开关）。
60. 新增 `scripts/run_lesson.py`：统一运行 lesson（track/lesson_id 参数）。
61. 更新 `docs/RUNNING.md`：增加如何运行 tracks。
62. 增强 CI：允许不装 torch 也通过（默认只跑 numpy 单测）；增加可选 torch job（可手动触发）。
63. 为每个 vision lesson 加 `checkpoints/` 与 `outputs/` 规范说明。
64. Vision 轨验收：`make smoke` 能跑通 MNIST lesson（不下载则用 compact 模式）。

### M4：NLP 轨（65–76）

65. 创建 `tracks/nlp/README.md`：明确旧 TF/Keras 内容将被 PyTorch 重写。
66. NLP Lesson 01：文本分类（IMDb/AGNews 或 compact），统一数据管线与训练。
67. NLP Lesson 02：词向量/Embedding 与 OOV 处理（简化实现）。
68. NLP Lesson 03：RNN/GRU baseline（对照 transformer）。
69. NLP Lesson 04：Attention 机制（从零写 scaled dot-product）。
70. NLP Lesson 05：Transformer encoder（小模型、compact 任务）。
71. NLP Lesson 06：NER（把 `Deep_project/ner` 思路迁移到 torch）。
72. NLP Lesson 07：阅读理解（把 BiDAF 思路做“简化实现”，不追求 SOTA）。
73. 把 `keras_text_classification` 的内容转成“对照阅读页”。
74. 为 nlp 轨添加 tokenizer 最小实现（不强依赖 huggingface）。
75. 为 NLP lessons 加快速 smoke（compact 输入，10 steps）。
76. 标记并准备删除 TF/Keras 旧脚本（删除前必须有新 lesson 跑通）。

### M5：GNN 轨（77–88）

77. 创建 `tracks/gnn/README.md`：图数据、消息传递、邻居聚合的统一解释。
78. GNN Lesson 01：Cora 数据加载与基础图操作（纯 torch sparse/edge_index）。
79. GNN Lesson 02：GCN（对照 `graph/pygcn`，重写现代风格）。
80. GNN Lesson 03：GAT（对照 `graph/GAT`，重写现代风格）。
81. GNN Lesson 04：GIN（对照 `graph/gin`，重写现代风格）。
82. GNN Lesson 05：GraphSAGE（不依赖原 graphsage 工程，最小实现）。
83. 给每个 GNN lesson 增加最小验收（loss/acc、运行时间上限）。
84. 增加 `tracks/gnn/tests/`：确保 forward shape 正确、无 NaN。
85. 把 `graph/label_propagation` 迁移为 “传统图方法补充课”。
86. 整理 `graph/*` 旧目录的“保留价值清单”（用于后续删除决策）。
87. 标记并准备删除旧 graph 工程（删除前必须有新 lesson + 说明文档）。
88. GNN 轨 smoke：能跑 1 个 epoch（或 50 steps）完成。

### M6：PointCloud 轨（89–96）

89. 创建 `tracks/pointcloud/README.md`：点云任务类型与数据格式说明。
90. PointCloud Lesson 01：PointNet 分类（最小实现，简化数据管线）。
91. PointCloud Lesson 02：PointNet 分割（简化实现，可选）。
92. PointCloud Lesson 03：DGCNN（简化实现，强调动态图构建与复杂度）。
93. 把 `Deep_project/Pointnet_Pointnet2` 代码拆解成“参考阅读页”，然后重写核心。
94. 为点云 lesson 增加 compact 点集数据（避免大数据下载）。
95. 增加 pointcloud 轨 smoke：compact 数据上能训练几步不报错。
96. 标记并准备删除旧点云工程（删除前有新 lesson + smoke）。

### M7：Generative 轨（97–100）

97. Generative Lesson 01：VAE（对照 `Deep_project/VAE`，统一结构重写）。
98. Generative Lesson 02：GAN（对照 `GAN/`，重写最小 DCGAN/MNIST）。
99. 生成模型 lesson 的可视化输出规范（保存 grid、固定噪声、对比图）。
100. 收敛清理：按 `docs/DELETE_LIST.md` 的“已替换清单”批量删除旧目录/旧脚本，并保留可复现验收命令作为证据。
