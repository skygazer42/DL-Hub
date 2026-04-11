# Transparent Object Segmentation

Toy-first local transparent object segmentation families for DL-Hub.

Available families:
- `glassseg_toy`
- `translab_seg`
- `refractmask_seg`
- `camotransparent_seg`
- `trimap_transparent`
- `boundary_glass_seg`
- `transformer_transparent`
- `diffusion_transparent`
- `prompt_transparent`
- `mamba_transparent`

Each family module exposes `build_<family>_transparent_segmenter(...)` with `tiny`, `small`, and `base` variants.
