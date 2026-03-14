# Diffusion Local Zoo

This directory mirrors the lightweight local zoo pattern used elsewhere in the repository.
Each diffusion family lives in its own Python module and exposes three local variants:

- `tiny`
- `small`
- `base`

Included families:

- `ddpm`
- `ddim`
- `iddpm`
- `score_sde`
- `ncsnpp`
- `edm`
- `latent_diffusion`
- `stable_diffusion`
- `consistency_model`
- `flow_matching`
- `rectified_flow`
- `conditional_flow_matching`

Helpers:

- `dlhub.generative.diffusion_zoo.list_local_arches()`
- `dlhub.generative.diffusion_zoo.build_local_model()`
- `python scripts/diffusion_zoo.py --list`
- `python scripts/diffusion_zoo.py --timeline`
- `python scripts/diffusion_zoo.py --recommend fidelity --variant tiny --top-k 4`
