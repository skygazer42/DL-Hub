# PointCloud 轨（点云）

目标：用 **toy-first** 的方式理解点云的“无序点集”特性，并把 PointNet/DGCNN 的核心直觉跑通：

- 点云是集合（set），不是序列：顺序不应该影响结果
- “局部邻域”与“全局汇聚”的差异
- 简单任务先闭环：toy 分类 → 再上更大数据与更复杂网络

## Lessons

- `lesson_01_pointnet_toy_classification/`：PointNet 分类（toy 数据：Cube vs Sphere）
- `lesson_02_dgcnn_toy_classification/`：DGCNN 分类（toy 数据：Cube vs Sphere）
- `lesson_03_pointnet2_toy_classification/`：PointNet2 分类（toy 数据：Cube vs Sphere）
