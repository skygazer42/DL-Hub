# Video Diffusion Local Zoo

This directory mirrors the lightweight local zoo pattern used by
`dlhub.generative.diffusion`, but specializes the family naming around video generation.
Each video diffusion family lives in its own Python module and exposes three local variants:

- `tiny`
- `small`
- `base`

Included families:

- `latent_video_diffusion`
- `frame_interp_diffusion`
- `video_unet_diffusion`
- `cascade_video_diffusion`
- `motion_prior_diffusion`
- `control_video_diffusion`
- `transformer_video_diffusion`
- `rectified_video_diffusion`
- `prompt_video_diffusion`
- `mamba_video_diffusion`

Helpers:

- `dlhub.generative.video_diffusion_zoo.list_local_arches()`
- `dlhub.generative.video_diffusion_zoo.build_local_model()`
- `python -c "from dlhub.generative.video_diffusion_zoo import list_local_arches; print(list_local_arches())"`
