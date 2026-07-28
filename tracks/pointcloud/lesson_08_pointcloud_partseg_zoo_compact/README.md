# Lesson 08：点云 partseg zoo（compact-first）

目标：像 `lesson_04_pointcloud_zoo_compact_classification` 一样，提供一个统一入口：

- 同一份小规模 part segmentation synthetic 数据
- 通过 `--arch` 快速切换 PointNet/DGCNN 两种思路

## 运行

列出 arch：

```bash
python -m tracks.pointcloud.lesson_08_pointcloud_partseg_zoo_compact.train --list-arch
```

PointNet：

```bash
python -m tracks.pointcloud.lesson_08_pointcloud_partseg_zoo_compact.train \
  --arch pointnet \
  --epochs 5
```

DGCNN（动态图 kNN）：

```bash
python -m tracks.pointcloud.lesson_08_pointcloud_partseg_zoo_compact.train \
  --arch dgcnn \
  --epochs 5
```

## 代码

- 数据：`tracks/pointcloud/synthetic_clouds.py`（`SyntheticPartSegDataset`）
- Model zoo：`tracks/pointcloud/lesson_08_pointcloud_partseg_zoo_compact/model.py`
