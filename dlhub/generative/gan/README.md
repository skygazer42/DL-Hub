# Generative GAN Local Zoo

This directory provides **24 local GAN families** with `tiny/small/base` variants
(72 architecture IDs in total).

- Arch ID format: `gan:<family>_<variant>`
- Unified builder: `dlhub.generative.gan_zoo.build_local_model(...)`
- CLI entry: `python scripts/gan_zoo.py`

## Quick Commands

```bash
python scripts/gan_zoo.py --list
python scripts/gan_zoo.py --timeline
python scripts/gan_zoo.py --list-profiles
python scripts/gan_zoo.py --recommend balanced --top-k 8 --variant tiny
python scripts/gan_zoo.py --smoke gan:dcgan_tiny
```

## Families by Group

| Group | Families |
|---|---|
| `vanilla_adversarial` | `dcgan`, `lsgan`, `wgan`, `wgangp`, `hingegan`, `relativistic_gan`, `dragan` |
| `conditional_gan` | `cgan`, `acgan`, `projection_gan`, `infogan`, `stackgan` |
| `image_translation` | `pix2pix`, `cyclegan`, `dualgan`, `unit`, `cutgan` |
| `high_fidelity` | `stylegan`, `stylegan2`, `stylegan3`, `biggan`, `sagan`, `progan`, `transgan` |
