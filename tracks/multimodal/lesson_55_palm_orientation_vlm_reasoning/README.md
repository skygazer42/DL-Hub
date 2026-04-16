# Lesson 55: Palm Orientation VLM Reasoning

This toy lesson regresses a normalized **palm orientation** target from a grayscale hand crop plus a
short text query. Each sample renders a palm-like blob with a thumb offset and finger ridge aligned
to a continuous orientation value in `[0, 1]`.

The lesson keeps the multimodal contract identical to the neighboring reasoning lessons while
switching the head and metrics to scalar regression.

Run:

```bash
python -m tracks.multimodal.lesson_55_palm_orientation_vlm_reasoning.train --device cpu --epochs 1
```

Outputs are written to `outputs/multimodal/lesson_55_palm_orientation_vlm_reasoning/<run_name>/`.
