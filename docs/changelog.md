# 更新日志

按功能模块分组，最新改动在前。

---

## 2026-07-28 · 实现语义与仓库叙事重构

- 课程与模型规模统一使用 `compact`，程序生成输入统一使用 `synthetic`，共享实现统一使用 `baseline`；维护代码不再保留旧的随意规模标签或双重兼容导入。
- 重命名 187 个课程目录与相关模块/API，并同步训练日志、测试、文档链接和运行示例。
- 新增[实现契约](implementation-contract.md)与 `scripts/narrative_check.py`，明确课程、数据、验证和 Model Zoo 审计之间的完整链路。
- 新增 Model Zoo 保真度台账；仅有通用基线的论文入口明确标为 `baseline-alias`，未审计实现保持 `unreviewed`。
- 移除 15 个旧规模标签注册别名后，本地 Zoo 注册 ID 实测为 8611；历史 8626 数字保留在下方版本记录中。
- 作者元数据统一为 `skygazer42 <207829897@qq.com>`；本地开发默认 `make verify` 后只运行与改动相关的最小测试。

---

## 2026-07-26 · Zoo 与杂项文档数字刷新

- 重建 `docs/zoo/index.md` 子系统总览（22 个子系统），CLI 全部改为真实存在的 `python scripts/<xxx>_zoo.py` 脚本。
- 刷新 Zoo 统计至当时实测值：Lessons 339 / Vision 791 / NLP 814 / Point Cloud 64 / VLM 210 (70 族) / GAN 132 (44 族) / Diffusion 96 (32 族) / Federated 228 (76 族)，本地 Zoo 架构 ID 合计 8626（2026-07-26 统计快照）。
- 新增 `docs/zoo/research-directions.md`，收录 22 批研究方向子领域的包路径明细。
- 修复 quickstart 中 3 个失效的 lesson 命令（gnn / nlp / foundations），测试文件数改为实测口径（2026-07 实测 409）。

---

## 2026-04-18 · 文档事实同步

- 刷新 README 与 docs/ 全站统计数字至实测值：Lessons 339 / 测试 393 / ML 算法 31 / Zoo 架构 8545。
- 校正 Vision backbone 791 / NLP backbone 814 / Pointcloud 赛道 36 Lessons。
- 详见 `docs/superpowers/specs/2026-04-18-docs-fact-sync-design.md`。

---

## Vision 赛道

### 图像去雨 (Deraining)

- 新增 Transformer 去雨模型家族（Restormer 等）
- 新增 DID-MDN、RCDNet 去雨器
- 新增 DDN、SPANet 去雨器
- 完善 lesson 10 去雨文档和算法覆盖表
- 修复小尺寸 reflect-pad 输入的边界检查

### 超分辨率 (Super-Resolution)

- 新增合成超分辨率 lesson 及数据管线
- 新增 Super-Resolution Zoo CLI
- 新增 SwinIR 风格轻量超分模型家族
- 新增残差超分模型家族（EDSR 等）
- 新增 SRCNN、FSRCNN 超分家族
- 添加超分辨率共享工具模块

### 目标检测 (Detection)

- 新增行人检测 preset 系列：Deformable DETR, YOLOv10, YOLOv9, PP-YOLOE, RTMDet, NanoDet, EfficientDet, CenterNet, RT-DETR, YOLOX, YOLOv8, YOLOv5, SSD, Faster R-CNN, RetinaNet, FCOS
- 新增行人检测经典方法：ACF, DPM, HOG+SVM
- 新增 NMS / Soft-NMS / DIoU-NMS / WBF 后处理
- 新增合成行人检测 FCOS lesson
- 扩展 Detection Zoo 时间线和模型家族至 50+
- 新增 YOLOv1 合成检测 lesson

### 其他视觉任务

- 新增局部 Co-Segmentation Zoo
- 新增 Face Parsing / Style Transfer / Video Summarization Zoo
- 新增 Action Recognition Zoo
- 新增 Fine-Grained Recognition Zoo
- 新增 Lane Detection Zoo 及模型家族
- 新增 MOT (多目标跟踪) Zoo
- 扩展去噪 lesson：盲点去噪、真实噪声模型、30+ 去噪模型家族
- 新增全景分割 Zoo (40 个家族)
- 新增语义分割 + 实例分割算法集

### 视觉 Backbone

- 新增 30+ Backbone Zoo（SKNet, ResNeSt, Res2Net, PVT, ECA/CBAM-ResNet, MLP-Mixer 等）

---

## Multimodal 赛道

- 新增 16 课多模态教学赛道
- 新增 VLM Zoo (BLIP, LLaVA, PaLiGemma, Flamingo, Perceiver 等)
- 扩展 VLM Zoo 第二批
- 新增 Video VLM、Grounding、Mask Grounding 等 lesson
- 新增 2D-TAN / Multi-scale 2D-TAN 视频定位 lesson
- 新增 BMN 时序提议 lesson

---

## Generative 赛道

- 新增 Diffusion Zoo (DDPM, Score-based 等)
- 新增 GAN Zoo

---

## LLM 赛道

- 新增 Paper-inspired LLM 实现（多批次）
- 扩展 Educational Model Notes

---

## Point Cloud 赛道

- 新增 3D Detection Zoo (40 个家族)
- 新增 3D Semantic Segmentation Zoo (40 个家族)
- 新增 3D Instance Segmentation Zoo (30 个家族)
- 新增 Tracking3D Zoo
- 新增点云自监督学习：I-JEPA, MSN, Data2Vec, ReSSL
- 新增 30+ 点云 Backbone Zoo

---

## Federated Learning

- 新增 Federated Learning Strategy Zoo

---

## ML 算法

- 新增扩展算法包：Lasso, Elastic Net, Kernel Ridge, Gaussian Process, Kernel PCA, MDS, LLE, KDE
- 规范化已有算法代码风格

---

## 基础设施

- CI 改进：GitHub Actions 流水线优化
- 测试覆盖率提升至 126+ 测试文件
- 冒烟测试全面覆盖
- 文档站 MkDocs Material 配置
- Zoo CLI 统一约定和 smoke-all 检查
