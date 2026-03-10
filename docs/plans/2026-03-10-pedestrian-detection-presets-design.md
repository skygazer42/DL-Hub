# 行人检测 Presets（Detection Zoo）设计说明

**日期**：2026-03-10  
**范围**：DL-Hub / `dlhub.vision.detection_zoo`（本地模型 zoo，纯 torch，toy-first）

## 背景

仓库已经包含大量检测器家族（`dlhub/vision/detection/*.py`），并通过 `dlhub.vision.detection_zoo` 提供：

- `python scripts/detection_zoo.py --list`：列出可用 arch id
- `python scripts/detection_zoo.py --smoke <ARCH_ID>`：随机输入做前向 smoke

本需求希望在不引入数据集、不做完整训练闭环的前提下，新增“行人检测算法”入口，便于快速 smoke 与统一命名检索。

## 目标（Goals）

1. 新增 **8 个**“行人检测 presets”架构 id，可被 `--search pedestrian` 搜到。
2. 每个 preset 都能：
   - `python scripts/detection_zoo.py --smoke dldet:<arch>` 前向通过；
   - 在测试中做最小 backward smoke（`loss.backward()`）通过。
3. 新增实现尽量复用仓库现有检测器家族（风险低、代码少、易维护）。

## 非目标（Non-goals）

- 不实现传统行人检测（HOG+SVM / ACF / DPM / Haar 等）。
- 不做真实数据集训练、评测、mAP 计算或后处理（NMS/解码/匹配）。
- 不保证与原论文/官方实现一致；遵循仓库“toy-first”定位。

## 方案选择

采用“**preset（别名/封装）**”方案：为行人检测提供语义化 arch id，但底层复用已有 detector 家族实现。

理由：

- 复用现有实现（例如 `FCOS/RetinaNet/Faster R-CNN/SSD/YOLO/RT-DETR`），新增代码 minimal；
- 与 `detection_zoo` 的自动发现机制天然兼容（`_VARIANTS` + `build_*_detector`）；
- 每个 preset 独立文件，便于开 8 个分支并行开发、最后合并。

## 新增 Arch IDs（8 个）

统一命名为 `dldet:pedestrian_<family>`（全部小写）：

- `dldet:pedestrian_fcos`
- `dldet:pedestrian_retinanet`
- `dldet:pedestrian_faster_rcnn`
- `dldet:pedestrian_ssd`
- `dldet:pedestrian_yolov5`
- `dldet:pedestrian_yolov8`
- `dldet:pedestrian_yolox`
- `dldet:pedestrian_rtdetr`

说明：

- “pedestrian” 只是语义入口，**不限制** `num_classes`；推荐 smoke 用 `--num-classes 1`。
- 每个 arch id 底层映射到对应家族的 `*_tiny` 变体（计算量小，适合快速验证）。

## 代码结构设计

每个 preset 一个文件，放在 `dlhub/vision/detection/` 下：

- `dlhub/vision/detection/pedestrian_fcos.py`
- `dlhub/vision/detection/pedestrian_retinanet.py`
- `dlhub/vision/detection/pedestrian_faster_rcnn.py`
- `dlhub/vision/detection/pedestrian_ssd.py`
- `dlhub/vision/detection/pedestrian_yolov5.py`
- `dlhub/vision/detection/pedestrian_yolov8.py`
- `dlhub/vision/detection/pedestrian_yolox.py`
- `dlhub/vision/detection/pedestrian_rtdetr.py`

每个文件约定：

- 定义 `_VARIANTS`，且只包含 1 个 key（例如 `pedestrian_fcos`）。
- 暴露工厂函数 `build_pedestrian_<family>_detector(...)`，内部用 `build_aliased_detector` 转发到基础 builder。
- 提供 `__main__` smoke（复用 `_aliases.smoke_aliased_detector`），便于手动运行。

依赖：

- 复用 `dlhub/vision/detection/_aliases.py` 的 `build_aliased_detector` / `smoke_aliased_detector`
- 复用对应基础 detector builder（例如 `build_fcos_detector`）

`detection_zoo` 发现机制说明（无需改动）：

- `dlhub/vision/detection_zoo.py` 会扫描 `dlhub/vision/detection/*.py`，
  解析 `_VARIANTS` 的 key + `build_*_detector` 函数名，生成 `dldet:<variant>` arch id。

## 测试设计

新增一个专用 smoke 测试文件（torch 可选依赖，缺失时跳过）：

- `tests/test_dlhub_vision_pedestrian_detection_presets.py`

覆盖内容：

- 用 `dlhub.vision.detection_zoo.build_local_model("dldet:pedestrian_...")` 构建 8 个模型；
- 随机输入 `x = torch.randn(2, 3, 64, 64)` 做 forward；
- 将输出（tensor/dict/list）汇总成标量 loss，并 `loss.backward()`；
- 断言 loss 是 finite。

## 命令行验收（Manual acceptance）

- 列出 presets：
  - `python scripts/detection_zoo.py --list --search pedestrian`
- Smoke 单个 preset（示例）：
  - `python scripts/detection_zoo.py --smoke dldet:pedestrian_yolov8 --num-classes 1 --image-size 128`

## Git 工作流（8 分支 -> 1 整合分支）

目标工作流：

1. 建一个整合分支：`feat/pedestrian-detection-presets`
2. 建 8 个实现分支（每个分支只加 1 个 preset 文件）：
   - `feat/pedestrian-fcos`、`feat/pedestrian-retinanet`、...（共 8 个）
3. 合并到整合分支后，再补测试文件并跑 `pytest -q`。

备注：

- 为避免文件冲突，测试文件只在整合分支添加。
