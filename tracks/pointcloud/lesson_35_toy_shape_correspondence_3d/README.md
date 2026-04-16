# Lesson 35: Toy Shape Correspondence (3D PointCloud)

This lesson builds a small synthetic task for dense correspondence between two
point clouds with known one-to-one matches.

## What it covers

- deterministic toy source shapes (cube/sphere surfaces)
- rigid transform + permutation to create target shapes
- per-point correspondence target indices
- shape-correspondence model wiring using
  `dlhub.pointcloud.shape_correspondence_3d.fmnet_corr3d`
- simple cross-entropy training and accuracy tracking

## Run

```bash
python -m tracks.pointcloud.lesson_35_toy_shape_correspondence_3d.train \
  --epochs 1 \
  --max-train-batches 2 \
  --max-eval-batches 1 \
  --device cpu
```
