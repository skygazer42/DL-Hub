# Vision Detection 50 Algorithms Design

**Context**

`dlhub/vision/detection/` already contains 42 compact-first object detection families, but it does not yet provide:

- a broader, more historically complete detector family archive
- timeline metadata spanning `2014-2026`
- stable category metadata for grouped browsing
- CLI support for viewing detector evolution by year and family type

The goal for this expansion is to add **50 more real detector families** while preserving the existing repo conventions:

- one detector family per file under `dlhub/vision/detection/`
- `_VARIANTS` defined in each family module
- `build_*_detector(...)` builder per module
- `if __name__ == "__main__":` forward/backward smoke
- zero external detection-framework dependency
- compact-first implementations that emphasize forward structure and stable gradients over benchmark fidelity

## Scope

This rollout adds **50 new detection families** on top of the current 42. The archive remains focused on **real, published algorithm families**, with emphasis on **mainstream and influential methods from 2019-2025**. The time axis still spans `2014-2026`, but `2026` is not artificially filled with speculative entries.

Families are grouped into five user-facing categories:

- `single_stage`
- `two_stage`
- `keypoint_anchor_free`
- `transformer_query`
- `open_vocabulary_multimodal`

## New Families

### Single-stage

- `overfeat`
- `ron`
- `refinedet`
- `m2det`
- `yolov4`
- `ppyolo`
- `scaled_yolov4`
- `ppyolov2`
- `giraffedet`
- `yolov6`
- `yolov7`
- `damo_yolo`
- `gold_yolo`
- `yolov9`
- `yolov10`

### Two-stage

- `rcnn`
- `sppnet`
- `fast_rcnn`
- `hypernet`
- `tridentnet`
- `libra_rcnn`
- `grid_rcnn`
- `guided_anchoring_rcnn`
- `detectors`
- `dynamic_rcnn`

### Keypoint / Anchor-free

- `densebox`
- `unitbox`
- `point_linking_network`
- `borderdet`
- `autoassign`
- `centernet2`

### Transformer / Query-based

- `anchor_detr`
- `smca_detr`
- `yolos`
- `adamixer`
- `efficient_detr`
- `deta`
- `h_detr`
- `co_detr`
- `group_detr`
- `ddq_detr`

### Open-vocabulary / Multimodal

- `vild`
- `regionclip`
- `glip`
- `detclip`
- `owl_vit`
- `grounding_dino`
- `detclipv2`
- `yolo_world`
- `decola`

## Architecture

The repo keeps the existing file-per-family convention. Instead of introducing a second detector package, the expansion uses:

- lightweight shared helpers in `dlhub/vision/detection/_common.py`
- a new metadata source of truth in `dlhub/vision/detection/_timeline.py`
- existing lazy export and AST-based discovery in `dlhub/vision/detection/__init__.py` and `dlhub/vision/detection_zoo.py`

This preserves current tooling while making the archive easier to browse and test.

The implementation reuses a small set of common detector patterns:

- FPN-based single-stage conv detectors
- proposal-style two-stage detectors
- center / point / border / set-based anchor-free detectors
- query-decoder transformer detectors
- multimodal detectors with optional text-conditioned inputs

Open-vocabulary families support **optional text conditioning** through `text_tokens` or `class_embeddings`, but smoke mode must still run without external text encoders or downloads.

## Metadata and CLI

The detection zoo will gain a timeline metadata module similar to action recognition and FGVC:

- `year`
- `family`
- `method`
- `group`
- `reference`

`scripts/detection_zoo.py` will support `--timeline` and print grouped year-by-year output with example `arch_id`s.

This metadata layer is intentionally separate from model construction so future README tables and docs can reuse the same source of truth.

## Testing Strategy

The rollout stays test-first:

- add failing tests for timeline metadata and CLI output
- extend zoo smoke coverage to include representative new families from all five categories
- keep AST convention checks enforcing `_VARIANTS`, builder, and `__main__` smoke
- use focused pytest runs during each batch, then wider detection verification

Because the current working tree already contains substantial in-progress detection edits, implementation will stay in the current workspace rather than a fresh git worktree. Creating a clean worktree from `HEAD` would discard the active uncommitted detection state and would not reflect the code the user is currently editing.
