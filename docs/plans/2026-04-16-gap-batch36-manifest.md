# Batch 36 Track-Fill Manifest

Batch 36 starts immediately after the Batch 35 merge-back and preserves the fixed `10 worktree +
10 lane` execution loop. This round extends the LLM structured-output block into TOML and markdown
tables, pushes the generative editing/fusion arc into stroke-mask editing and layout-palette
fusion, continues multimodal hand understanding with palm-orientation and sign-digit reasoning,
advances the vision hand track into finger-spread and thumb-position estimation, and keeps the NLP
support-ops line moving with reopen and resolution-owner prediction.

Selection rules locked for this batch:

- fixed `10 worktree + 10 lane` execution
- track-first coverage across `tracks/llm`, `tracks/generative`, `tracks/multimodal`,
  `tracks/vision`, and `tracks/nlp`
- max 2 lanes per surface
- nearest teaching-neighbor fill after the current end of each track

Start point:

- base branch: `main`
- branch anchor commit: `docs: mark batch35 gaps merged`
- integration branch: `batch360-integration`
- worktree root: `.worktrees/`

## Lane Table

| lane_id | branch | worktree | surface | slug | lane_owned_paths | integration_owned_paths | verification_cmds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| L01 | `batch360-toml-constrained-prompting` | `.worktrees/batch360-toml-constrained-prompting` | `tracks/llm` | `toml_constrained_prompting_llm_lesson` | `tracks/llm/lesson_40_toy_toml_constrained_prompting/`, `tests/test_tracks_llm_toml_constrained_prompting.py` | `README.md`, `tracks/llm/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_llm_toml_constrained_prompting.py`<br>`python scripts/run_lesson.py llm lesson_40_toy_toml_constrained_prompting --dry-run` |
| L02 | `batch360-markdown-table-constrained-prompting` | `.worktrees/batch360-markdown-table-constrained-prompting` | `tracks/llm` | `markdown_table_prompting_llm_lesson` | `tracks/llm/lesson_41_toy_markdown_table_constrained_prompting/`, `tests/test_tracks_llm_markdown_table_constrained_prompting.py` | `README.md`, `tracks/llm/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_llm_markdown_table_constrained_prompting.py`<br>`python scripts/run_lesson.py llm lesson_41_toy_markdown_table_constrained_prompting --dry-run` |
| L03 | `batch360-stroke-mask-editing` | `.worktrees/batch360-stroke-mask-editing` | `tracks/generative` | `stroke_mask_editing_generative_lesson` | `tracks/generative/lesson_41_toy_diffusion_stroke_mask_editing/`, `tests/test_tracks_generative_stroke_mask_editing.py` | `README.md`, `tracks/generative/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_generative_stroke_mask_editing.py`<br>`python scripts/run_lesson.py generative lesson_41_toy_diffusion_stroke_mask_editing --dry-run` |
| L04 | `batch360-layout-palette-fusion` | `.worktrees/batch360-layout-palette-fusion` | `tracks/generative` | `layout_palette_fusion_generative_lesson` | `tracks/generative/lesson_42_toy_diffusion_layout_palette_fusion/`, `tests/test_tracks_generative_layout_palette_fusion.py` | `README.md`, `tracks/generative/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_generative_layout_palette_fusion.py`<br>`python scripts/run_lesson.py generative lesson_42_toy_diffusion_layout_palette_fusion --dry-run` |
| L05 | `batch360-palm-orientation-vlm` | `.worktrees/batch360-palm-orientation-vlm` | `tracks/multimodal` | `palm_orientation_multimodal_lesson` | `tracks/multimodal/lesson_55_palm_orientation_vlm_reasoning/`, `tests/test_tracks_multimodal_palm_orientation_reasoning.py` | `README.md`, `tracks/multimodal/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_multimodal_palm_orientation_reasoning.py`<br>`python scripts/run_lesson.py multimodal lesson_55_palm_orientation_vlm_reasoning --dry-run` |
| L06 | `batch360-sign-digit-vlm` | `.worktrees/batch360-sign-digit-vlm` | `tracks/multimodal` | `sign_digit_multimodal_lesson` | `tracks/multimodal/lesson_56_sign_digit_vlm_reasoning/`, `tests/test_tracks_multimodal_sign_digit_reasoning.py` | `README.md`, `tracks/multimodal/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_multimodal_sign_digit_reasoning.py`<br>`python scripts/run_lesson.py multimodal lesson_56_sign_digit_vlm_reasoning --dry-run` |
| L07 | `batch360-finger-spread-vision` | `.worktrees/batch360-finger-spread-vision` | `tracks/vision` | `finger_spread_estimation_vision_lesson` | `tracks/vision/lesson_56_synthetic_finger_spread_estimation/`, `tests/test_tracks_vision_finger_spread_estimation.py` | `README.md`, `tracks/vision/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_vision_finger_spread_estimation.py`<br>`python scripts/run_lesson.py vision lesson_56_synthetic_finger_spread_estimation --dry-run` |
| L08 | `batch360-thumb-position-vision` | `.worktrees/batch360-thumb-position-vision` | `tracks/vision` | `thumb_position_classification_vision_lesson` | `tracks/vision/lesson_57_synthetic_thumb_position_classification/`, `tests/test_tracks_vision_thumb_position_classification.py` | `README.md`, `tracks/vision/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_vision_thumb_position_classification.py`<br>`python scripts/run_lesson.py vision lesson_57_synthetic_thumb_position_classification --dry-run` |
| L09 | `batch360-dialog-reopen-prediction` | `.worktrees/batch360-dialog-reopen-prediction` | `tracks/nlp` | `dialog_reopen_prediction_nlp_lesson` | `tracks/nlp/lesson_46_toy_dialog_reopen_prediction/`, `tests/test_tracks_nlp_dialog_reopen_prediction.py` | `README.md`, `tracks/nlp/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_nlp_dialog_reopen_prediction.py`<br>`python scripts/run_lesson.py nlp lesson_46_toy_dialog_reopen_prediction --dry-run` |
| L10 | `batch360-dialog-resolution-owner` | `.worktrees/batch360-dialog-resolution-owner` | `tracks/nlp` | `dialog_resolution_owner_prediction_nlp_lesson` | `tracks/nlp/lesson_47_toy_dialog_resolution_owner_prediction/`, `tests/test_tracks_nlp_dialog_resolution_owner_prediction.py` | `README.md`, `tracks/nlp/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_nlp_dialog_resolution_owner_prediction.py`<br>`python scripts/run_lesson.py nlp lesson_47_toy_dialog_resolution_owner_prediction --dry-run` |

## Integration Branch Ownership

`batch360-integration` owns only the shared surfaces required after the 10 lane merges:

- `README.md`
- `tracks/llm/README.md`
- `tracks/generative/README.md`
- `tracks/multimodal/README.md`
- `tracks/vision/README.md`
- `tracks/nlp/README.md`
- `tests/test_scripts_run_lesson.py`
- any shared verification-only updates required to make the merged track batch pass on `main`

## Merge Order

1. `batch360-toml-constrained-prompting`
2. `batch360-markdown-table-constrained-prompting`
3. `batch360-stroke-mask-editing`
4. `batch360-layout-palette-fusion`
5. `batch360-palm-orientation-vlm`
6. `batch360-sign-digit-vlm`
7. `batch360-finger-spread-vision`
8. `batch360-thumb-position-vision`
9. `batch360-dialog-reopen-prediction`
10. `batch360-dialog-resolution-owner`

## Exit Criteria

- each lane has at least one local commit on its own branch
- each lane passes its manifest smoke commands from its own worktree
- integration updates the top-level lesson counts and track lesson tables for all 10 new lessons
- integration branch passes fresh shared verification before merge back to `main`
