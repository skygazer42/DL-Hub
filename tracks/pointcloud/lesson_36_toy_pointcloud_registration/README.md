# Lesson 36: Toy Point Cloud Registration

This lesson builds a small synthetic task for rigid registration between a source
point cloud and a transformed target point cloud.

## What it covers

- deterministic toy source shapes (cube/sphere surfaces)
- rigid pose targets with translation and Euler-angle supervision
- point-cloud registration wiring using `dlhub.pointcloud.registration.pointnetlk`
- compact MSE training and pose error tracking

## Run

```bash
python -m tracks.pointcloud.lesson_36_toy_pointcloud_registration.train \
  --epochs 1 \
  --max-train-batches 2 \
  --max-eval-batches 1 \
  --device cpu
```

