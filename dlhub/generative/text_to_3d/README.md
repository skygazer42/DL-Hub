# Text-to-3D Local Zoo

This directory mirrors the lightweight local zoo layout used by the other
generative directions in the repository.

Each family lives in its own module and exposes three local variants:

- `tiny`
- `small`
- `base`

Included families:

- `dreamfusion_toy`
- `magic3d_toy`
- `score_distill_3d`
- `neural_lift_3d`
- `sdf_prompt_3d`
- `mesh_diffuse_3d`
- `transformer_text3d`
- `gaussian_text3d`
- `layout_text3d`
- `mamba_text3d`

Helpers:

- `dlhub.generative.text_to_3d_zoo.list_local_arches()`
- `dlhub.generative.text_to_3d_zoo.build_local_model()`
