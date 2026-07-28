# Lesson 52: Gesture VLM Reasoning

This compact lesson classifies a compact gesture state from grayscale image evidence plus a short text
query. Each sample renders a simple stick-figure upper body whose arm configuration maps to one of
four gesture classes:

- `rest`
- `left_wave`
- `right_wave`
- `hands_up`

The goal is to keep the multimodal training loop CPU friendly while exposing a clear batch
contract and a minimal fusion model.

Run:

```bash
python -m tracks.multimodal.lesson_52_gesture_vlm_reasoning.train --device cpu --epochs 1
```

Outputs are written to `outputs/multimodal/lesson_52_gesture_vlm_reasoning/<run_name>/`.

