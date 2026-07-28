# Point Cloud Completion

Compact-first point cloud completion families for partial-to-complete shape prediction.

Included families:

- `pcn_completion`
- `topnet_completion`
- `grnet_completion`
- `snowflake_completion`
- `folding_completion`
- `anchor_completion`
- `transformer_completion`
- `diffusion_completion`
- `text_guided_completion`
- `mamba_completion`

Each module exposes `build_<family>_completer(...)` and defines `tiny`, `small`, and `base`
variants in a local `_VARIANTS` registry.

Example:

```python
from dlhub.pointcloud.pointcloud_completion.pcn_completion import (
    build_pcn_completion_completer,
)

model = build_pcn_completion_completer(in_channels=3, variant="pcn_completion_tiny")
```

The local zoo lives in `dlhub.pointcloud.pointcloud_completion_zoo` and uses the `pccomp:`
prefix.
