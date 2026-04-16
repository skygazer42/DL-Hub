# Lesson 83: Synthetic Panoptic Segmentation (Toy, CPU-friendly)

This lesson adds a minimal panoptic segmentation training loop on synthetic scenes.

- synthetic data with stuff background + thing rectangles
- semantic labels plus instance masks/classes per sample
- model wired to the local `dlhub.vision.panoptic_segmentation` family via
  `build_panoptic_fpn_panoptic_segmenter`
- compact loss: semantic CE + instance class CE + mask BCE

Run:

```bash
python -m tracks.vision.lesson_83_synthetic_panoptic_segmentation.train \
  --epochs 1 \
  --max-train-batches 2 \
  --max-eval-batches 1 \
  --device cpu
```

