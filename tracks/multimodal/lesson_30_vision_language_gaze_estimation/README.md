# Lesson 30: Vision-Language Gaze Estimation (Toy)

This toy lesson predicts a gaze target point and heatmap from:

- image features,
- normalized head location cues, and
- a short directional language prompt.

It is intentionally small and CPU-friendly, designed to demonstrate multimodal data flow
and basic supervision for gaze prediction.

Run:

```bash
python -m tracks.multimodal.lesson_30_vision_language_gaze_estimation.train --device cpu --epochs 1
```
