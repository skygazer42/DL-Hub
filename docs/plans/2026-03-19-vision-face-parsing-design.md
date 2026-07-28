# Vision Face Parsing Design

**Goal**

Add a local, compact-first face parsing algorithm family to DL-Hub with:
- local face parsing model families
- a unified zoo + CLI
- fast smoke tests that run on CPU with no downloads

This follows the repository patterns used by:
- `dlhub.vision.segmentation`
- `dlhub.vision.instance_segmentation_zoo`
- `dlhub.vision.video_summarization_zoo`

---

## Scope

This feature targets face parsing rather than generic scene segmentation.

Input:
- image tensor with shape `(B, C, H, W)`

Output:
- `logits`: face-part logits with shape `(B, K, H, W)`
- `parsing_map`: argmax parsing map with shape `(B, H, W)`

Optional outputs may include:
- `binary_edge`
- `category_edge`
- `roi_attention`
- `implicit_features`

The family is meant to teach how specialized face parsing methods combine local facial priors, boundary reasoning, and lightweight dense prediction heads, not to exactly reproduce benchmark systems.

---

## Package Layout

Add:
- `dlhub/vision/face_parsing/`
- `dlhub/vision/face_parsing_zoo.py`
- `scripts/face_parsing_zoo.py`
- `tests/test_dlhub_vision_face_parsing_zoo.py`

Initial local families:
- `roi_tanh_warp`: RoI Tanh-Warping style local-global face parser
- `dml_csr`: decoupled multi-task face parser with edge-aware refinement
- `fp_liif`: local implicit image function face parser
- `stn_icnn`: spatial-transformer coarse-to-fine face parser
- `segface`: semantic-gated multi-scale face parser
- `facexformer_parse`: part-query transformer face parser
- `occlusion_tanh`: occlusion-aware tanh-transform face parser
- `mask_fpan`: de-occlusion and UV-guided face parser
- `farl_parse`: visual-linguistic prompt-aligned face parser
- `eagrnet`: edge-aware graph reasoning face parser
- `agrnet`: adaptive graph representation learning face parser
- `ehanet`: hierarchical aggregation face parser

Each family provides three variants:
- `*_tiny`
- `*_small`
- `*_base`

Zoo prefix:
- `fparse:<variant>`

Current expanded local coverage:
- 12 families
- 36 arches

---

## Model Contract

Every family exposes:
- `_VARIANTS`
- `build_<family>_face_parser(...)`

Every model supports:
- `model(image)` where `image` is `(B, C, H, W)`

Every model returns a dict with:
- `logits`
- `parsing_map`

---

## Testing

Add focused tests for:
- listing arches
- building representative families
- random-image forward smoke
- CLI `--list`
- CLI `--smoke`
