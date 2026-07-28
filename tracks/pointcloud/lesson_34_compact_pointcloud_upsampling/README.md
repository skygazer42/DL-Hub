# Lesson 34: Compact Pointcloud Upsampling

This lesson demonstrates sparse-to-dense pointcloud upsampling with a tiny
synthetic dataset and a lightweight upsampler from
`dlhub.pointcloud.pointcloud_upsampling`.

## What You Learn

- how to construct deterministic sparse/dense pointcloud pairs
- how to plug a first-class upsampling family into a track lesson
- how to train and evaluate using Chamfer distance

## Run

```bash
python -m tracks.pointcloud.lesson_34_compact_pointcloud_upsampling.train \
  --epochs 3 \
  --arch punet_upsample:punet_upsample_tiny \
  --num-sparse-points 64 \
  --upsample-factor 2
```
