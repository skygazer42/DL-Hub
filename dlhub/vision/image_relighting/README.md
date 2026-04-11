# Image Relighting

Toy-first local image relighting families for DL-Hub.

Available families:
- `deep_relight`
- `hdr_relight`
- `intrinsic_relight`
- `ratio_relight`
- `retinex_relight`
- `portrait_relight`
- `transformer_relight`
- `diffusion_relight`
- `prompt_relight`
- `mamba_relight`

Each family module exposes `build_<family>_relighter(...)` with `tiny`, `small`, and `base` variants.
