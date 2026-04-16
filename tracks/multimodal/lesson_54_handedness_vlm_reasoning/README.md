# Lesson 54: Handedness VLM Reasoning

This toy lesson classifies **handedness** (`left` vs `right`) from grayscale image evidence plus a
short text query. Each sample renders a simple hand-like blob where a "thumb" bump appears on the
left or right side of the palm.

The goal is to keep the multimodal training loop CPU friendly while exposing a clear batch
contract and a minimal fusion model.

Run:

```bash
python -m tracks.multimodal.lesson_54_handedness_vlm_reasoning.train --device cpu --epochs 1
```

Outputs are written to `outputs/multimodal/lesson_54_handedness_vlm_reasoning/<run_name>/`.
