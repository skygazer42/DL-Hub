# Vision Co-Segmentation Design

**Goal**

Add a local, toy-first co-segmentation algorithm family to DL-Hub with:
- local co-segmentation model families
- a unified zoo + CLI
- fast smoke tests that run on CPU with no downloads

This follows the repository patterns used by:
- `dlhub.vision.segmentation`
- `dlhub.vision.instance_segmentation_zoo`
- `dlhub.vision.face_parsing_zoo`

---

## Scope

This feature targets group-based image co-segmentation rather than ordinary single-image segmentation.

Input:
- image set tensor with shape `(B, T, C, H, W)`

Output:
- `logits`: co-segmentation logits with shape `(B, T, K, H, W)`
- `masks`: argmax masks with shape `(B, T, H, W)`

Optional outputs may include:
- `co_attention`
- `group_tokens`
- `prototype_masks`
- `consensus_map`

The family is meant to teach how co-segmentation methods combine shared-image encoding, group-level correspondence, consensus modeling, and lightweight dense prediction heads, not to exactly reproduce benchmark systems.

---

## Package Layout

Add:
- `dlhub/vision/co_segmentation/`
- `dlhub/vision/co_segmentation_zoo.py`
- `scripts/co_segmentation_zoo.py`
- `tests/test_dlhub_vision_co_segmentation_zoo.py`

Initial local families:
- `siamese_coseg`: shared-weight image encoder with group matching
- `cosal_uformer`: co-saliency-style U-shaped group aggregator
- `group_proto_net`: group prototype mining and broadcast refinement
- `co_attention_fpn`: multi-scale FPN fusion with co-attention
- `transformer_coseg`: token-level image-set transformer co-segmentor
- `consensus_refiner`: coarse independent masks plus group consensus refinement

Each family provides three variants:
- `*_tiny`
- `*_small`
- `*_base`

Zoo prefix:
- `coseg:<variant>`

Current local coverage:
- 6 families
- 18 arches

---

## Model Contract

Every family exposes:
- `_VARIANTS`
- `build_<family>_co_segmentor(...)`

Every model supports:
- `model(images)` where `images` is `(B, T, C, H, W)`

Every model returns a dict with:
- `logits`
- `masks`

---

## Shared Components

`dlhub/vision/co_segmentation/_common.py` should provide:
- `check_btchw`
- `logits_to_masks`
- `flatten_group`
- `unflatten_group`
- `TinyCoSegEncoder`
- `GroupFusionBlock`
- `CoSegHead`

The shared code should keep the first batch of families CPU-friendly and consistent while still making each algorithm family structurally distinct.

---

## Testing

Add focused tests for:
- listing arches
- building representative families
- random image-set forward smoke
- CLI `--list`
- CLI `--search`
- CLI `--smoke`

The canonical smoke tensor should use:
- `images.shape == (2, 3, 3, 64, 64)`

Core output assertions:
- `logits.shape == (2, 3, K, 64, 64)`
- `masks.shape == (2, 3, 64, 64)`
