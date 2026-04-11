# Event Camera Understanding

Toy-first event camera understanding families for local experimentation.

Families in this package:

- `ev_cnn`
- `voxel_eventnet`
- `spike_eventnet`
- `event_unet`
- `event_tracker`
- `event_depth`
- `transformer_event`
- `state_space_event`
- `crossmodal_event`
- `mamba_event`

Each family exposes `build_<family>_event_model(...)` with `tiny`, `small`, and `base` variants.
