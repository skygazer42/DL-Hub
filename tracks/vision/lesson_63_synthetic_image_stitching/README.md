# Lesson 63: Synthetic Image Stitching

This toy vision lesson renders a compact synthetic panorama, then exposes two overlapping partial
views that must be stitched back into the full image. The setup stays CPU-friendly while still
capturing the core structure of classic image stitching: incomplete coverage, overlap, and a
single fused panorama target.

Run:

```bash
python -m tracks.vision.lesson_63_synthetic_image_stitching.train --epochs 1 --device cpu
```

Outputs land under `outputs/vision/lesson_63_synthetic_image_stitching/<run_name>/`.
