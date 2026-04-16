# Batch 16 Completion Manifest

Batch 16 is the one approved half-batch exception. It completes the five remaining direction lanes
that were already designed in the 2026-04-14 batch-16 design/plan pair.

Start point:

- base branch: `main`
- branch anchor commit: `plan: seed whole-repo gap register and batch16 manifest`
- integration branch: `batch160-integration`
- worktree root: `.worktrees/`

## Lane Table

| lane_id | branch | worktree | surface | slug | lane_owned_paths | integration_owned_paths | verification_cmds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| L01 | `batch160-embodied-question-answering` | `.worktrees/batch160-embodied-question-answering` | `dlhub/multimodal` | `embodied_question_answering` | `dlhub/multimodal/embodied_question_answering/`, `dlhub/multimodal/embodied_question_answering_zoo.py` | `README.md` | `python -c "from dlhub.multimodal.embodied_question_answering.navqa_embodied import build_navqa_embodied_embodied_qa_model as f; print(type(f(variant='navqa_embodied_tiny')).__name__)"`<br>`python -c "from dlhub.multimodal.embodied_question_answering_zoo import list_local_arches; print(any('navqa_embodied_tiny' in x for x in list_local_arches()))"` |
| L02 | `batch160-audio-text-understanding` | `.worktrees/batch160-audio-text-understanding` | `dlhub/multimodal` | `audio_text_understanding` | `dlhub/multimodal/audio_text_understanding/`, `dlhub/multimodal/audio_text_understanding_zoo.py` | `README.md` | `python -c "from dlhub.multimodal.audio_text_understanding.audio_bert_understanding import build_audio_bert_understanding_audio_text_model as f; print(type(f(variant='audio_bert_understanding_tiny')).__name__)"`<br>`python -c "from dlhub.multimodal.audio_text_understanding_zoo import list_local_arches; print(any('audio_bert_understanding_tiny' in x for x in list_local_arches()))"` |
| L03 | `batch160-text-to-video` | `.worktrees/batch160-text-to-video` | `dlhub/generative` | `text_to_video` | `dlhub/generative/text_to_video/`, `dlhub/generative/text_to_video_zoo.py` | `README.md` | `python -c "from dlhub.generative.text_to_video.zeroscope_t2v import build_zeroscope_t2v_text_to_video as f; print(type(f(variant='zeroscope_t2v_tiny')).__name__)"`<br>`python -c "from dlhub.generative.text_to_video_zoo import list_local_arches; print(any('zeroscope_t2v_tiny' in x for x in list_local_arches()))"` |
| L04 | `batch160-video-to-video` | `.worktrees/batch160-video-to-video` | `dlhub/generative` | `video_to_video` | `dlhub/generative/video_to_video/`, `dlhub/generative/video_to_video_zoo.py` | `README.md` | `python -c "from dlhub.generative.video_to_video.vid2vid_translation import build_vid2vid_translation_video_to_video as f; print(type(f(variant='vid2vid_translation_tiny')).__name__)"`<br>`python -c "from dlhub.generative.video_to_video_zoo import list_local_arches; print(any('vid2vid_translation_tiny' in x for x in list_local_arches()))"` |
| L05 | `batch160-world-models` | `.worktrees/batch160-world-models` | `dlhub/generative` | `world_models` | `dlhub/generative/world_models/`, `dlhub/generative/world_models_zoo.py` | `README.md` | `python -c "from dlhub.generative.world_models.rssm_world import build_rssm_world_world_model as f; print(type(f(variant='rssm_world_tiny')).__name__)"`<br>`python -c "from dlhub.generative.world_models_zoo import list_local_arches; print(any('rssm_world_tiny' in x for x in list_local_arches()))"` |

## Integration Branch Ownership

`batch160-integration` owns only shared surfaces:

- `README.md`
- any shared package exports needed after branch merges
- any cross-direction smoke or targeted pytest additions needed to make Batch 16 verifiable on `main`

## Merge Order

1. `batch160-embodied-question-answering`
2. `batch160-audio-text-understanding`
3. `batch160-text-to-video`
4. `batch160-video-to-video`
5. `batch160-world-models`

## Exit Criteria

- each lane has at least one local commit
- each lane passes its manifest smoke commands
- integration branch restores the README batch-16 section to 10/10 directions
- integration branch passes fresh shared verification before merge back to `main`
