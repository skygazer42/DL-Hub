# Vision Style Transfer (Classic NST + Translation) Design

**Goal**

Add a new "style transfer" algorithm family to DL-Hub with:
- Local algorithm implementations (toy-first, no downloads)
- A unified local zoo + CLI for listing and smoke-running models
- Two new Vision lessons: classic neural style transfer and translation-based style transfer

This should match existing repository conventions:
- One family per file, exposing `build_*` factory functions
- A `*_zoo.py` that can list/build local architectures without importing everything eagerly
- A `scripts/*_zoo.py` CLI with `--list` and `--smoke`
- Minimal, fast tests that instantiate models and run forward (optionally backward)

---

## Package Layout

Add new modules:
- `dlhub/vision/style_transfer/`
  - `*_*.py`: one algorithm family per file
  - `_common.py`: shared toy building blocks (enc/dec, discriminator, style ops)
- `dlhub/vision/style_transfer_zoo.py`: discovery + build wrapper
- `scripts/style_transfer_zoo.py`: CLI for `--list` and `--smoke`

Add new lessons:
- `tracks/vision/lesson_15_neural_style_transfer_gatys/`
  - Optimization-based NST (Gatys-style), toy-first, CPU smoke-friendly
- `tracks/vision/lesson_16_style_transfer_translation_cyclegan/`
  - Cycle-consistency image translation (CycleGAN-style), toy-first, CPU smoke-friendly

Update documentation:
- `tracks/vision/README.md` to list the two new lessons.

---

## Local Zoo API

`dlhub.vision.style_transfer_zoo` exposes:
- `list_local_arches() -> list[str]` returning namespaced ids like `dlst:adain_tiny`
- `build_local_model(arch_id, *, in_channels, image_size, width_mult, dropout, **kwargs) -> nn.Module`

Style transfer models expose a unified forward interface:
- `model(content, style)` returns a `dict[str, Tensor]` with at least:
  - `stylized`: `(B, C, H, W)`

Translation-based families may return additional tensors (`fake_a`, `fake_b`, logits, etc.)
but must always include `stylized`.

---

## Algorithm Families (First Batch)

Classic NST:
- `gatys`: content/style loss with Gram matrices (optimization-based)
- `fast_nst`: Johnson-style feed-forward residual stylizer
- `adain`: feature statistics alignment (arbitrary style)
- `wct`: whitening-color transform (arbitrary style)

Translation-based:
- `pix2pix`: paired translation (A->B)
- `cyclegan`: unpaired cycle-consistency (A<->B)
- `cut`: unpaired translation with patch contrast (toy simplification)
- `munit`: content/style disentanglement (toy simplification)

Each family provides 3 variants: `*_tiny`, `*_small`, `*_base`.

---

## Algorithm Families (Second Batch, Popular 2018-2022)

Arbitrary style transfer:
- `avatar_net`: Avatar-Net-style feature decoration (toy: WCT + local attention refinement)
- `sanet`: SANet-style attention (toy cross-attention in feature space)
- `stytr2`: Transformer style transfer (toy cross-attention blocks)

Translation / reference-conditioned:
- `ugatit`: U-GAT-IT-inspired attention + AdaLIN (toy, reference-conditioned)
- `starganv2`: StarGAN v2-inspired reference-conditioned translation (toy)

These are implemented under the same conventions:
- `model(content, style)` returns a dict with at least `stylized`.
- Each family provides 3 variants: `*_tiny`, `*_small`, `*_base`.

---

## Algorithm Families (Third Batch, Diffusion-Based 2022-2026)

These families are diffusion-inspired / Stable Diffusion ecosystem concepts, implemented as tiny local
models (no pretrained checkpoints required):

- `stylediffusion`: latent diffusion img2img conditioned on a style reference embedding
- `controlnet`: ControlNet-style structural hint (edges) conditioning during denoising
- `ip_adapter`: IP-Adapter-style image-prompt cross-attention conditioning
- `cfg_stylediffusion`: classifier-free guidance (CFG) conditioning variant (style reference as condition)
- `style_aligned`: self-attention + reference-attention denoising (toy "style-aligned" idea)

Additional arbitrary style transfer:
- `adaattn`: AdaAttN-style per-position style statistics via attention

Additional diffusion / editing-style families:
- `sdedit`: SDEdit-style noisy img2img editing toward a style reference latent
- `instantstyle`: InstantStyle-style decoupled style-token + global style-code injection
- `attenst`: AttenST-style attention-driven stylization with content-aware adaptive normalization

They follow the same interface:
- `model(content, style)` returns a dict with at least `stylized`.
- Variants: `*_tiny`, `*_small`, `*_base`.
