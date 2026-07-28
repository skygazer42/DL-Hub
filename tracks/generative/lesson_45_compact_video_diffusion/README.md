# Lesson 45: Compact Video Diffusion

This lesson defines a compact conditional video diffusion task. The dataset synthesizes a short
grayscale clip of a blob moving according to a low-dimensional motion code, while the model learns
to denoise a noised target clip given the first-frame keyframe plus the motion code.

Batch contract:

- `keyframe`: `(B, 1, H, W)` float tensor in `[0, 1]`
- `motion_code`: `(B, 3)` float tensor containing `(dx, dy, brightness)`
- `target_video`: `(B, 1, T, H, W)` float tensor in `[0, 1]`

Run:

```bash
python -m tracks.generative.lesson_45_compact_video_diffusion.train --epochs 1 --device cpu
```

Outputs are written under `outputs/generative/lesson_45_compact_video_diffusion/<run_name>/`.
