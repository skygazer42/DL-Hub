# Lesson 42: Compact Dialog Resolution Time Prediction

This lesson classifies whether a short in-progress support dialog is likely to resolve quickly,
in a standard time window, or late. The synthetic corpus keeps cues explicit so the classifier
stays CPU friendly.

Each example includes tokens like `resolution`, `time`, and `minutes` plus light context about
channel, issue type, mood, and agent state. Labels are three buckets: `quick`, `standard`, and
`late`.

Run:

```bash
python -m tracks.nlp.lesson_42_compact_dialog_resolution_time_prediction.train --device cpu --epochs 1
```

Outputs are written to
`outputs/nlp/lesson_42_compact_dialog_resolution_time_prediction/<run_name>/`.
