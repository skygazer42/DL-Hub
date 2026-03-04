# Lesson 12 — Vision 目标检测：YOLOv1-style（toy-first）

目标：在合成数据上跑通一个 **YOLOv1 风格**的最小目标检测闭环：

- 单张图只包含 **1 个矩形目标**
- 输出为网格（stride=4）的：
  - `obj_logits`：是否有物体
  - `cls_logits`：类别（toy 里只有 1 类）
  - `bbox`：归一化 `cx, cy, w, h`

## 运行

从 repo 根目录运行：

```bash
python -m tracks.vision.lesson_12_synthetic_detection_yolo.train --run-name dev
```

