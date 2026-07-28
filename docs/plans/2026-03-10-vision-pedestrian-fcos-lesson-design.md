# Vision Pedestrian Detection (Synthetic FCOS) — Design

## Summary

Add a new **Vision** lesson that trains a lightweight **FCOS-style** detector on an offline,
synthetic “pedestrian” dataset (tall, slender rectangles). The lesson uses the repository’s
local detection zoo preset `dldet:pedestrian_fcos` so users can directly reuse the newly added
pedestrian presets.

This is intentionally **compact-first** and focuses on the full training loop (data → model → loss →
metrics → checkpoint), not production-grade decoding/NMS.

## Goals

- Provide an **end-to-end** runnable pedestrian-detection example (CPU-friendly).
- Require **no downloads**: data is procedurally generated.
- Reuse the local detection zoo preset:
  - default arch id: `dldet:pedestrian_fcos`
- Maintain repo conventions:
  - `tracks/vision/lesson_xx_*/` layout
  - outputs written via `dlhub.paths.build_run_paths`
  - metrics logged to `metrics.jsonl`

## Non-Goals

- No full COCO-style dataloaders, augmentation pipelines, or real datasets.
- No NMS / multi-box decoding / AP metrics.
- No training support for all 8 pedestrian presets in one script (their outputs differ).

## Dataset Design

- **Input:** synthetic RGB images `(3, H, W)` with background noise + a single “pedestrian”:
  - pedestrian = bright tall rectangle (height > width), optionally with a small “head” blob
  - random location and size in a reasonable range
- **Targets:** a single positive cell on a stride-4 grid:
  - `cls_target`: `(1, Gh, Gw)` one-hot at the person center cell
  - `reg_target`: `(4, Gh, Gw)` l/t/r/b distances (only filled at the positive cell)
  - `centerness_target`: `(1, Gh, Gw)` FCOS-style centerness computed from l/t/r/b (only filled at the positive cell)
  - `pos_mask`: `(1, Gh, Gw)` one-hot mask for gathering regression targets
  - `box`: `(4,)` ground-truth `xyxy` in pixel coordinates (for IoU metric)

This mirrors the existing `lesson_04_synthetic_detection_fcos` design but switches to RGB and
adds centerness training to match the default `fcos_tiny` preset behavior.

## Model + Training

- Model is built via `dlhub.vision.detection_zoo.build_local_model`:
  - `arch_id="dldet:pedestrian_fcos"`
  - `in_channels=3`, `num_classes=1`, `width_mult` configurable
- Losses (simple, stable):
  - `cls_loss`: `BCEWithLogitsLoss(pos_weight=...)` on `cls_logits`
  - `reg_loss`: `SmoothL1Loss` on gathered `ltrb` at the positive cell
  - `center_loss`: `BCEWithLogitsLoss(pos_weight=...)` on `centerness` (if present)
- Metrics:
  - `center_acc`: argmax cell matches target cell
  - `mean_iou`: decode 1 bbox per image (best cell) and compute IoU to GT

## CLI + Outputs

`python -m tracks.vision.lesson_13_synthetic_pedestrian_detection_fcos.train ...`

Key args:
- data: `--num-samples --batch-size --image-size --stride --noise-std --min-box-w/h --max-box-w/h`
- training: `--epochs --learning-rate --device --max-train-batches --max-eval-batches --run-name`
- model: `--arch --width-mult`

Outputs:
- `outputs/vision/lesson_13_synthetic_pedestrian_detection_fcos/<run_name>/`
  - `config.json`
  - `metrics.jsonl`
  - `logs/train.log`
  - `checkpoints/checkpoint.pt`

## Testing Strategy

- Add a fast pytest smoke that:
  - builds dataloader with tiny settings
  - builds the model via detection zoo (`dldet:pedestrian_fcos`)
  - runs a forward pass and computes a finite loss
  - validates output/target shapes

## Future Extensions (Optional)

- Add a second lesson to demonstrate a YOLO-style pedestrian preset with its own target format.
- Add a lightweight visualization utility to draw predicted bbox on the synthetic image grid.

