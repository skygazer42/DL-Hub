# Vision Deraining Expansion Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Expand `tracks/vision/lesson_10_synthetic_denoising` with six additional toy-first deraining families, wire them into lesson discovery, and document the enlarged deraining cluster.

**Architecture:** Keep deraining inside the existing denoising lesson rather than creating a new package or lesson. Each new family lives in its own file under `dlhub/vision/denoising/`, follows the existing `build_*_denoiser(...)` factory convention, and is surfaced through `tracks/vision/lesson_10_synthetic_denoising/model.py` so it automatically participates in lesson discovery and training.

**Tech Stack:** Python, PyTorch, argparse, pytest

---

### Task 1: Add the failing deraining regression tests

**Files:**
- Modify: `F:/DL-Hub/tests/test_tracks_vision_denoising.py`

**Step 1: Write the failing rain-model smoke test**

Add a new focused test such as:

```python
@pytest.mark.parametrize(
    "arch",
    [
        "ddn:ddn_tiny",
        "spanet:spanet_tiny",
        "did_mdn:did_mdn_tiny",
        "rcdnet:rcdnet_tiny",
        "transweather:transweather_tiny",
        "derainformer:derainformer_tiny",
    ],
)
def test_vision_deraining_rain_models_forward_backward_smoke(arch: str) -> None:
    from tracks.vision.lesson_10_synthetic_denoising.data import DataConfig, get_dataloaders
    from tracks.vision.lesson_10_synthetic_denoising.model import ModelConfig, build_model

    train_loader, _ = get_dataloaders(
        DataConfig(
            num_samples=64,
            batch_size=4,
            image_size=32,
            val_fraction=0.2,
            seed=0,
            num_workers=0,
            in_channels=1,
            noise_type="rain",
            noise_std=0.15,
            rain_count=24,
            rain_length_min=8,
            rain_length_max=18,
            rain_intensity_min=0.05,
            rain_intensity_max=0.14,
            min_square=6,
            max_square=10,
            train_mode="supervised",
        )
    )
    rainy, clean = next(iter(train_loader))
    model = build_model(ModelConfig(arch=arch, variant="", in_channels=1, sigma=0.15))
    pred = model(rainy)
    assert tuple(pred.shape) == tuple(clean.shape)
    loss = torch.nn.L1Loss()(pred, clean)
    assert torch.isfinite(loss)
    loss.backward()
```

**Step 2: Extend discovery assertions so they mention the new families**

Update existing CLI / listing tests to assert that outputs include:
- `ddn`
- `spanet`
- `did_mdn`
- `rcdnet`
- `transweather`
- `derainformer`

At minimum, extend:
- `test_vision_denoising_train_parse_args_list_arch_families`
- one `--list-arch` or `--arch-family` regression test

**Step 3: Run the focused tests to verify they fail**

Run:
- `python -m pytest tests/test_tracks_vision_denoising.py -q -k "deraining_rain_models_forward_backward_smoke or list_arch_families or arch_filters_by_family"`

Expected:
- failures because the new families are not implemented or not wired

**Step 4: Commit**

```bash
git add tests/test_tracks_vision_denoising.py
git commit -m "test: add deraining expansion regression coverage"
```

### Task 2: Implement `ddn` and `spanet`

**Files:**
- Create: `F:/DL-Hub/dlhub/vision/denoising/ddn.py`
- Create: `F:/DL-Hub/dlhub/vision/denoising/spanet.py`
- Modify: `F:/DL-Hub/tracks/vision/lesson_10_synthetic_denoising/model.py`

**Step 1: Run the focused tests for the first two families**

Run:
- `python -m pytest tests/test_tracks_vision_denoising.py -q -k "ddn or spanet"`

Expected:
- fail because builders and dispatch branches do not exist yet

**Step 2: Write the minimal `ddn` implementation**

Implement a small detail-restoration derainer:

```python
class DDN(nn.Module):
    def __init__(self, *, in_channels: int, width: int, depth: int) -> None:
        ...
        self.detail = nn.Conv2d(in_channels, width, kernel_size=3, padding=1)
        self.body = nn.Sequential(*[ResidualBlock(width) for _ in range(depth)])
        self.head = nn.Conv2d(width, in_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        detail = x - F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        feat = self.body(F.relu(self.detail(detail), inplace=True))
        rain = self.head(feat)
        return x - rain
```

Add `_VARIANTS` with `ddn_tiny`, `ddn_small`, `ddn_base`, and expose
`build_ddn_denoiser(...)`.

**Step 3: Write the minimal `spanet` implementation**

Implement a compact spatial-attention derainer:

```python
class SPANet(nn.Module):
    def __init__(self, *, in_channels: int, width: int, depth: int) -> None:
        ...
        self.stem = nn.Conv2d(in_channels, width, kernel_size=3, padding=1)
        self.body = nn.Sequential(*[ResidualBlock(width) for _ in range(depth)])
        self.attn = nn.Conv2d(width, 1, kernel_size=3, padding=1)
        self.head = nn.Conv2d(width, in_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.body(F.relu(self.stem(x), inplace=True))
        mask = torch.sigmoid(self.attn(feat))
        rain = self.head(feat) * mask
        return x - rain
```

Add `_VARIANTS` and `build_spanet_denoiser(...)`.

**Step 4: Wire `ddn` and `spanet` into lesson model dispatch**

Modify `tracks/vision/lesson_10_synthetic_denoising/model.py` so:
- `list_supported_arches()` includes `ddn:*` and `spanet:*`
- `build_model()` supports `ddn` and `spanet`

**Step 5: Re-run the focused tests**

Run:
- `python -m pytest tests/test_tracks_vision_denoising.py -q -k "ddn or spanet"`

Expected:
- `ddn` and `spanet` cases pass
- the other new deraining families still fail

**Step 6: Commit**

```bash
git add dlhub/vision/denoising/ddn.py dlhub/vision/denoising/spanet.py tracks/vision/lesson_10_synthetic_denoising/model.py
git commit -m "feat: add ddn and spanet derainers"
```

### Task 3: Implement `did_mdn` and `rcdnet`

**Files:**
- Create: `F:/DL-Hub/dlhub/vision/denoising/did_mdn.py`
- Create: `F:/DL-Hub/dlhub/vision/denoising/rcdnet.py`
- Modify: `F:/DL-Hub/tracks/vision/lesson_10_synthetic_denoising/model.py`

**Step 1: Run the focused tests for `did_mdn` and `rcdnet`**

Run:
- `python -m pytest tests/test_tracks_vision_denoising.py -q -k "did_mdn or rcdnet"`

Expected:
- fail because builders and dispatch branches do not exist yet

**Step 2: Write the minimal `did_mdn` implementation**

Implement a small multi-branch dense derainer:

```python
class DIDMDN(nn.Module):
    def __init__(self, *, in_channels: int, width: int, growth: int, branches: int) -> None:
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.stem(x)
        branch_feats = [branch(feat) for branch in self.branches]
        fused = self.fuse(torch.cat(branch_feats, dim=1))
        rain = self.head(fused)
        return x - rain
```

Keep the public return value as a tensor, not a dict, so it matches the lesson
contract.

**Step 3: Write the minimal `rcdnet` implementation**

Implement a small unrolled decomposition network:

```python
class RCDNet(nn.Module):
    def __init__(self, *, in_channels: int, width: int, stages: int) -> None:
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        clean = x
        rain = torch.zeros_like(x)
        for _ in range(self.stages):
            rain = self.rain_block(torch.cat([clean, x], dim=1))
            clean = x - rain
            clean = clean + self.clean_block(clean)
        return clean
```

Add `_VARIANTS` and `build_rcdnet_denoiser(...)`.

**Step 4: Wire both families into lesson model dispatch**

Modify `tracks/vision/lesson_10_synthetic_denoising/model.py` so:
- `list_supported_arches()` includes `did_mdn:*` and `rcdnet:*`
- `build_model()` supports the aliases `did_mdn`, `did-mdn`, `didmdn`
- `build_model()` supports `rcdnet`

**Step 5: Re-run the focused tests**

Run:
- `python -m pytest tests/test_tracks_vision_denoising.py -q -k "did_mdn or rcdnet"`

Expected:
- `did_mdn` and `rcdnet` cases pass

**Step 6: Commit**

```bash
git add dlhub/vision/denoising/did_mdn.py dlhub/vision/denoising/rcdnet.py tracks/vision/lesson_10_synthetic_denoising/model.py
git commit -m "feat: add did-mdn and rcdnet derainers"
```

### Task 4: Implement `transweather` and `derainformer`

**Files:**
- Create: `F:/DL-Hub/dlhub/vision/denoising/transweather.py`
- Create: `F:/DL-Hub/dlhub/vision/denoising/derainformer.py`
- Modify: `F:/DL-Hub/tracks/vision/lesson_10_synthetic_denoising/model.py`

**Step 1: Run the focused transformer-family tests**

Run:
- `python -m pytest tests/test_tracks_vision_denoising.py -q -k "transweather or derainformer"`

Expected:
- fail because these families are missing

**Step 2: Write the minimal `transweather` implementation**

Implement a small patch-transformer restoration model using the existing
padding helpers in `dlhub/vision/denoising/_utils.py` if needed:

```python
class TransWeather(nn.Module):
    def __init__(self, *, in_channels: int, embed_dim: int, depth: int, num_heads: int) -> None:
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        padded, pad_hw = pad_to_multiple(x, 4)
        feat = self.patch_embed(padded)
        tokens = self.encoder(feat.flatten(2).transpose(1, 2))
        feat = tokens.transpose(1, 2).reshape(...)
        rain = self.head(feat)
        out = padded - rain
        return unpad(out, pad_hw)
```

**Step 3: Write the minimal `derainformer` implementation**

Implement a hybrid conv-transformer derainer:

```python
class DerainFormer(nn.Module):
    def __init__(self, *, in_channels: int, embed_dim: int, depth: int, num_heads: int) -> None:
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.stem(x)
        tokens = self.to_tokens(feat)
        tokens = self.blocks(tokens)
        feat = self.to_map(tokens, x.shape[-2:])
        rain = self.head(feat)
        return x - rain
```

Keep tiny variants compact enough for CPU smoke tests.

**Step 4: Wire both families into lesson model dispatch**

Modify `tracks/vision/lesson_10_synthetic_denoising/model.py` so:
- `list_supported_arches()` includes `transweather:*` and `derainformer:*`
- `build_model()` supports these family names and light aliases

**Step 5: Re-run the focused tests**

Run:
- `python -m pytest tests/test_tracks_vision_denoising.py -q -k "transweather or derainformer"`

Expected:
- transformer-family cases pass

**Step 6: Commit**

```bash
git add dlhub/vision/denoising/transweather.py dlhub/vision/denoising/derainformer.py tracks/vision/lesson_10_synthetic_denoising/model.py
git commit -m "feat: add transformer deraining families"
```

### Task 5: Update package exports and lesson documentation

**Files:**
- Modify: `F:/DL-Hub/dlhub/vision/denoising/__init__.py`
- Modify: `F:/DL-Hub/tracks/vision/lesson_10_synthetic_denoising/README.md`

**Step 1: Update package exports**

Add the new classes and builders to:
- top-level imports
- `__all__`

The module should export:
- `DDN`, `build_ddn_denoiser`
- `SPANet`, `build_spanet_denoiser`
- `DIDMDN`, `build_did_mdn_denoiser`
- `RCDNet`, `build_rcdnet_denoiser`
- `TransWeather`, `build_transweather_denoiser`
- `DerainFormer`, `build_derainformer_denoiser`

**Step 2: Update the lesson README**

Revise the deraining-related sections so they mention the enlarged cluster and
example commands such as:

```bash
python -m tracks.vision.lesson_10_synthetic_denoising.train \
  --arch ddn:ddn_tiny --noise-type rain --epochs 3

python -m tracks.vision.lesson_10_synthetic_denoising.train \
  --arch transweather:transweather_tiny --noise-type rain --epochs 3
```

Also extend the “模型一览” deraining table to cover all nine deraining families:
- `jorder`
- `rescan`
- `prenet`
- `ddn`
- `spanet`
- `did_mdn`
- `rcdnet`
- `transweather`
- `derainformer`

**Step 3: Re-run the documentation-adjacent test slice**

Run:
- `python -m pytest tests/test_tracks_vision_denoising.py -q -k "list_arch_families or arch_filters_by_family or arch_filters_by_match"`

Expected:
- listing tests pass with the new families included

**Step 4: Commit**

```bash
git add dlhub/vision/denoising/__init__.py tracks/vision/lesson_10_synthetic_denoising/README.md
git commit -m "docs: expand lesson 10 deraining coverage"
```

### Task 6: Run final verification

**Files:**
- Verify only

**Step 1: Run the focused denoising suite**

Run:
- `python -m pytest tests/test_tracks_vision_denoising.py -q`

Expected:
- the full denoising test file passes

**Step 2: Run representative lesson CLI discovery commands**

Run:
- `python -m tracks.vision.lesson_10_synthetic_denoising.train --list-arch-families`
- `python -m tracks.vision.lesson_10_synthetic_denoising.train --list-arch --arch-family ddn --list-limit 5`
- `python -m tracks.vision.lesson_10_synthetic_denoising.train --list-arch --arch-family transweather --list-limit 5`

Expected:
- family names print cleanly
- the two family-filtered commands show the new variants

**Step 3: Run representative rain-mode lesson smoke commands**

Run:
- `python -m tracks.vision.lesson_10_synthetic_denoising.train --arch ddn:ddn_tiny --noise-type rain --device cpu --epochs 1 --num-samples 128 --batch-size 16 --max-train-batches 1 --max-eval-batches 1 --run-name derain_ddn_smoke`
- `python -m tracks.vision.lesson_10_synthetic_denoising.train --arch transweather:transweather_tiny --noise-type rain --device cpu --epochs 1 --num-samples 128 --batch-size 8 --max-train-batches 1 --max-eval-batches 1 --run-name derain_transweather_smoke`

Expected:
- both runs complete
- outputs are written under `outputs/vision/lesson_10_synthetic_denoising/<run_name>/`

**Step 4: Perform completion verification**

Use `@verification-before-completion` and verify:
- fresh test and CLI commands were run
- outputs match the claimed status
- no missing wiring remains between the new family files and lesson discovery

**Step 5: Commit**

```bash
git add dlhub/vision/denoising/__init__.py dlhub/vision/denoising/ddn.py dlhub/vision/denoising/spanet.py dlhub/vision/denoising/did_mdn.py dlhub/vision/denoising/rcdnet.py dlhub/vision/denoising/transweather.py dlhub/vision/denoising/derainformer.py tracks/vision/lesson_10_synthetic_denoising/model.py tracks/vision/lesson_10_synthetic_denoising/README.md tests/test_tracks_vision_denoising.py
git commit -m "feat: expand lesson 10 deraining families"
```
