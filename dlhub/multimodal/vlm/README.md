# VLM Local Zoo

This directory provides a local, toy-first Vision-Language Model zoo.

Families in the first batch:

- 2021: `vilt`, `clip`, `align`, `albef`
- 2022: `ofa`, `blip`, `coca`, `flamingo`
- 2023: `blip2`, `instructblip`, `llava`, `kosmos2`

Second batch extensions:

- 2021: `simvlm`, `lit`
- 2022: `pali`
- 2023: `pali_x`, `minigpt4`, `mplug_owl2`, `qwen_vl`, `cogvlm`

Each family exposes three local variants:

- `tiny`
- `small`
- `base`

Helpers:

- `dlhub.multimodal.vlm_zoo.list_local_arches()`
- `dlhub.multimodal.vlm_zoo.build_local_model()`
- `python scripts/vlm_zoo.py --list`
- `python scripts/vlm_zoo.py --timeline`
- `python scripts/vlm_zoo.py --recommend instruction --variant tiny --top-k 4`
