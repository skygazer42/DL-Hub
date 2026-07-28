# Batch 35 Track-Fill Manifest

Batch 35 starts immediately after the Batch 34 merge-back and preserves the fixed `10 worktree +
10 lane` expansion loop. This round extends the LLM structured-output block into YAML and CSV
constraints, pushes the generative editing/fusion arc into scribble-mask editing and layout-style
fusion, continues multimodal hand understanding with finger-count and handedness reasoning,
advances the vision hand track into palm-orientation and sign-digit recognition, and keeps the NLP
support-ops line moving with SLA-breach and follow-up-channel prediction.

Selection rules locked for this batch:

- fixed `10 worktree + 10 lane` execution
- track-first coverage across `tracks/llm`, `tracks/generative`, `tracks/multimodal`,
  `tracks/vision`, and `tracks/nlp`
- max 2 lanes per surface
- nearest teaching-neighbor fill after the current end of each track

Start point:

- base branch: `main`
- branch anchor commit: `docs: mark batch34 gaps merged`
- integration branch: `batch350-integration`
- worktree root: `.worktrees/`

## Lane Table

| lane_id | branch | worktree | surface | slug | lane_owned_paths | integration_owned_paths | verification_cmds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| L01 | `batch350-yaml-constrained-prompting` | `.worktrees/batch350-yaml-constrained-prompting` | `tracks/llm` | `yaml_constrained_prompting_llm_lesson` | `tracks/llm/lesson_38_compact_yaml_constrained_prompting/`, `tests/test_tracks_llm_yaml_constrained_prompting.py` | `README.md`, `tracks/llm/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_llm_yaml_constrained_prompting.py`<br>`python scripts/run_lesson.py llm lesson_38_compact_yaml_constrained_prompting --dry-run` |
| L02 | `batch350-csv-constrained-prompting` | `.worktrees/batch350-csv-constrained-prompting` | `tracks/llm` | `csv_constrained_prompting_llm_lesson` | `tracks/llm/lesson_39_compact_csv_constrained_prompting/`, `tests/test_tracks_llm_csv_constrained_prompting.py` | `README.md`, `tracks/llm/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_llm_csv_constrained_prompting.py`<br>`python scripts/run_lesson.py llm lesson_39_compact_csv_constrained_prompting --dry-run` |
| L03 | `batch350-scribble-mask-editing` | `.worktrees/batch350-scribble-mask-editing` | `tracks/generative` | `scribble_mask_editing_generative_lesson` | `tracks/generative/lesson_39_compact_diffusion_scribble_mask_editing/`, `tests/test_tracks_generative_scribble_mask_editing.py` | `README.md`, `tracks/generative/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_generative_scribble_mask_editing.py`<br>`python scripts/run_lesson.py generative lesson_39_compact_diffusion_scribble_mask_editing --dry-run` |
| L04 | `batch350-layout-style-fusion` | `.worktrees/batch350-layout-style-fusion` | `tracks/generative` | `layout_style_fusion_generative_lesson` | `tracks/generative/lesson_40_compact_diffusion_layout_style_fusion/`, `tests/test_tracks_generative_layout_style_fusion.py` | `README.md`, `tracks/generative/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_generative_layout_style_fusion.py`<br>`python scripts/run_lesson.py generative lesson_40_compact_diffusion_layout_style_fusion --dry-run` |
| L05 | `batch350-finger-count-vlm` | `.worktrees/batch350-finger-count-vlm` | `tracks/multimodal` | `finger_count_multimodal_lesson` | `tracks/multimodal/lesson_53_finger_count_vlm_reasoning/`, `tests/test_tracks_multimodal_finger_count_reasoning.py` | `README.md`, `tracks/multimodal/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_multimodal_finger_count_reasoning.py`<br>`python scripts/run_lesson.py multimodal lesson_53_finger_count_vlm_reasoning --dry-run` |
| L06 | `batch350-handedness-vlm` | `.worktrees/batch350-handedness-vlm` | `tracks/multimodal` | `handedness_multimodal_lesson` | `tracks/multimodal/lesson_54_handedness_vlm_reasoning/`, `tests/test_tracks_multimodal_handedness_reasoning.py` | `README.md`, `tracks/multimodal/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_multimodal_handedness_reasoning.py`<br>`python scripts/run_lesson.py multimodal lesson_54_handedness_vlm_reasoning --dry-run` |
| L07 | `batch350-palm-orientation-vision` | `.worktrees/batch350-palm-orientation-vision` | `tracks/vision` | `palm_orientation_estimation_vision_lesson` | `tracks/vision/lesson_54_synthetic_palm_orientation_estimation/`, `tests/test_tracks_vision_palm_orientation_estimation.py` | `README.md`, `tracks/vision/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_vision_palm_orientation_estimation.py`<br>`python scripts/run_lesson.py vision lesson_54_synthetic_palm_orientation_estimation --dry-run` |
| L08 | `batch350-sign-digit-vision` | `.worktrees/batch350-sign-digit-vision` | `tracks/vision` | `sign_digit_classification_vision_lesson` | `tracks/vision/lesson_55_synthetic_sign_digit_classification/`, `tests/test_tracks_vision_sign_digit_classification.py` | `README.md`, `tracks/vision/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_vision_sign_digit_classification.py`<br>`python scripts/run_lesson.py vision lesson_55_synthetic_sign_digit_classification --dry-run` |
| L09 | `batch350-dialog-sla-breach` | `.worktrees/batch350-dialog-sla-breach` | `tracks/nlp` | `dialog_sla_breach_prediction_nlp_lesson` | `tracks/nlp/lesson_44_compact_dialog_sla_breach_prediction/`, `tests/test_tracks_nlp_dialog_sla_breach_prediction.py` | `README.md`, `tracks/nlp/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_nlp_dialog_sla_breach_prediction.py`<br>`python scripts/run_lesson.py nlp lesson_44_compact_dialog_sla_breach_prediction --dry-run` |
| L10 | `batch350-dialog-followup-channel` | `.worktrees/batch350-dialog-followup-channel` | `tracks/nlp` | `dialog_followup_channel_prediction_nlp_lesson` | `tracks/nlp/lesson_45_compact_dialog_followup_channel_prediction/`, `tests/test_tracks_nlp_dialog_followup_channel_prediction.py` | `README.md`, `tracks/nlp/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_nlp_dialog_followup_channel_prediction.py`<br>`python scripts/run_lesson.py nlp lesson_45_compact_dialog_followup_channel_prediction --dry-run` |

## Integration Branch Ownership

`batch350-integration` owns only the shared surfaces required after the 10 lane merges:

- `README.md`
- `tracks/llm/README.md`
- `tracks/generative/README.md`
- `tracks/multimodal/README.md`
- `tracks/vision/README.md`
- `tracks/nlp/README.md`
- `tests/test_scripts_run_lesson.py`
- any shared verification-only updates required to make the merged track batch pass on `main`

## Merge Order

1. `batch350-yaml-constrained-prompting`
2. `batch350-csv-constrained-prompting`
3. `batch350-scribble-mask-editing`
4. `batch350-layout-style-fusion`
5. `batch350-finger-count-vlm`
6. `batch350-handedness-vlm`
7. `batch350-palm-orientation-vision`
8. `batch350-sign-digit-vision`
9. `batch350-dialog-sla-breach`
10. `batch350-dialog-followup-channel`

## Exit Criteria

- each lane has at least one local commit on its own branch
- each lane passes its manifest smoke commands from its own worktree
- integration updates the top-level lesson counts and track lesson tables for all 10 new lessons
- integration branch passes fresh shared verification before merge back to `main`
