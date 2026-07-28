# Raindrop Removal Local Zoo

Compact-first raindrop removal models implemented in pure PyTorch.

## Included Families

- `mask_guided_drop`
- `streak_aware_drop`
- `texture_refine_drop`
- `dual_branch_drop`
- `recurrent_drop`
- `transformer_drop`
- `frequency_drop`
- `context_drop`
- `prompt_drop`
- `mamba_drop`

Each family provides `tiny`, `small`, and `base` variants.

## Quick Example

```python
from dlhub.vision.raindrop_removal_zoo import build_local_model

model = build_local_model("drop:transformer_drop_small", in_channels=3)
out = model(image)
restored = out["restored"]
mask = out["raindrop_mask"]
```
