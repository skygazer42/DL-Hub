# Lesson 57: Finger Spread VLM Reasoning

This compact lesson regresses a normalized **finger spread** target from a grayscale hand crop and a
short text query. Each sample renders a palm with four fingers whose separation changes with the
target spread value in `[0, 1]`.

The lesson follows the same multimodal reasoning contract as the neighboring hand VLM lessons while
using scalar regression metrics.

Run:

```bash
python -m tracks.multimodal.lesson_57_finger_spread_vlm_reasoning.train --device cpu --epochs 1
```

Outputs are written to `outputs/multimodal/lesson_57_finger_spread_vlm_reasoning/<run_name>/`.
