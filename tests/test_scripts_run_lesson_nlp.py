"""Tests for scripts/run_lesson.py: nlp track lessons."""

import subprocess
import sys

from _run_lesson_helpers import _repo_root


def test_run_lesson_dry_run_resolves_nlp_transformer_summarization_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "nlp",
            "lesson_09_toy_transformer_summarization",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.nlp.lesson_09_toy_transformer_summarization.train" in proc.stdout


def test_run_lesson_dry_run_resolves_nlp_prompt_tuning_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "nlp",
            "lesson_10_toy_prompt_tuning_classifier",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.nlp.lesson_10_toy_prompt_tuning_classifier.train" in proc.stdout


def test_run_lesson_dry_run_resolves_nlp_few_shot_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "nlp",
            "lesson_11_toy_few_shot_text_classification",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.nlp.lesson_11_toy_few_shot_text_classification.train" in proc.stdout


def test_run_lesson_dry_run_resolves_nlp_in_context_text_classification_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "nlp",
            "lesson_12_toy_in_context_text_classification",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.nlp.lesson_12_toy_in_context_text_classification.train" in proc.stdout


def test_run_lesson_dry_run_resolves_nlp_masked_language_modeling_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "nlp",
            "lesson_13_toy_masked_language_modeling",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.nlp.lesson_13_toy_masked_language_modeling.train" in proc.stdout


def test_run_lesson_dry_run_resolves_nlp_contrastive_sentence_embedding_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "nlp",
            "lesson_14_toy_contrastive_sentence_embedding",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.nlp.lesson_14_toy_contrastive_sentence_embedding.train" in proc.stdout


def test_run_lesson_dry_run_resolves_nlp_cross_encoder_reranking_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "nlp",
            "lesson_15_toy_cross_encoder_reranking",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.nlp.lesson_15_toy_cross_encoder_reranking.train" in proc.stdout


def test_run_lesson_dry_run_resolves_nlp_text_clustering_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "nlp",
            "lesson_16_toy_text_clustering",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.nlp.lesson_16_toy_text_clustering.train" in proc.stdout


def test_run_lesson_dry_run_resolves_nlp_text_anomaly_detection_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "nlp",
            "lesson_17_toy_text_anomaly_detection",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.nlp.lesson_17_toy_text_anomaly_detection.train" in proc.stdout


def test_run_lesson_dry_run_resolves_nlp_topic_modeling_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "nlp",
            "lesson_18_toy_topic_modeling",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.nlp.lesson_18_toy_topic_modeling.train" in proc.stdout


def test_run_lesson_dry_run_resolves_nlp_distilled_text_classifier_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "nlp",
            "lesson_19_toy_distilled_text_classifier",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.nlp.lesson_19_toy_distilled_text_classifier.train" in proc.stdout


def test_run_lesson_dry_run_resolves_nlp_adversarial_text_classification_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "nlp",
            "lesson_20_toy_adversarial_text_classification",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.nlp.lesson_20_toy_adversarial_text_classification.train" in proc.stdout


def test_run_lesson_dry_run_resolves_nlp_adversarial_example_detection_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "nlp",
            "lesson_21_toy_adversarial_example_detection",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.nlp.lesson_21_toy_adversarial_example_detection.train" in proc.stdout


def test_run_lesson_dry_run_resolves_nlp_weak_supervision_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "nlp",
            "lesson_22_toy_weak_supervision_text_classification",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.nlp.lesson_22_toy_weak_supervision_text_classification.train" in proc.stdout


def test_run_lesson_dry_run_resolves_nlp_sentence_denoising_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "nlp",
            "lesson_23_toy_sentence_denoising_autoencoder",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.nlp.lesson_23_toy_sentence_denoising_autoencoder.train" in proc.stdout


def test_run_lesson_dry_run_resolves_nlp_meta_few_shot_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "nlp",
            "lesson_24_toy_meta_few_shot_text_classification",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.nlp.lesson_24_toy_meta_few_shot_text_classification.train" in proc.stdout


def test_run_lesson_dry_run_resolves_nlp_low_shot_intent_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "nlp",
            "lesson_25_toy_low_shot_intent_detection",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.nlp.lesson_25_toy_low_shot_intent_detection.train" in proc.stdout


def test_run_lesson_dry_run_resolves_nlp_dialog_slot_prediction_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "nlp",
            "lesson_36_toy_dialog_slot_prediction",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.nlp.lesson_36_toy_dialog_slot_prediction.train" in proc.stdout


def test_run_lesson_dry_run_resolves_nlp_dialog_outcome_prediction_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "nlp",
            "lesson_37_toy_dialog_outcome_prediction",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.nlp.lesson_37_toy_dialog_outcome_prediction.train" in proc.stdout


def test_run_lesson_dry_run_resolves_nlp_dialog_satisfaction_prediction_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "nlp",
            "lesson_38_toy_dialog_satisfaction_prediction",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.nlp.lesson_38_toy_dialog_satisfaction_prediction.train" in proc.stdout


def test_run_lesson_dry_run_resolves_nlp_dialog_escalation_risk_prediction_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "nlp",
            "lesson_39_toy_dialog_escalation_risk_prediction",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.nlp.lesson_39_toy_dialog_escalation_risk_prediction.train" in proc.stdout


def test_run_lesson_dry_run_resolves_nlp_dialog_priority_prediction_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "nlp",
            "lesson_40_toy_dialog_priority_prediction",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.nlp.lesson_40_toy_dialog_priority_prediction.train" in proc.stdout


def test_run_lesson_dry_run_resolves_nlp_dialog_transfer_prediction_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "nlp",
            "lesson_41_toy_dialog_transfer_prediction",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.nlp.lesson_41_toy_dialog_transfer_prediction.train" in proc.stdout


def test_run_lesson_dry_run_resolves_nlp_dialog_resolution_time_prediction_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "nlp",
            "lesson_42_toy_dialog_resolution_time_prediction",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.nlp.lesson_42_toy_dialog_resolution_time_prediction.train" in proc.stdout


def test_run_lesson_dry_run_resolves_nlp_dialog_callback_prediction_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "nlp",
            "lesson_43_toy_dialog_callback_prediction",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.nlp.lesson_43_toy_dialog_callback_prediction.train" in proc.stdout


def test_run_lesson_dry_run_resolves_nlp_dialog_sla_breach_prediction_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "nlp",
            "lesson_44_toy_dialog_sla_breach_prediction",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.nlp.lesson_44_toy_dialog_sla_breach_prediction.train" in proc.stdout


def test_run_lesson_dry_run_resolves_nlp_dialog_followup_channel_prediction_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "nlp",
            "lesson_45_toy_dialog_followup_channel_prediction",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.nlp.lesson_45_toy_dialog_followup_channel_prediction.train" in proc.stdout


def test_run_lesson_dry_run_resolves_nlp_dialog_reopen_prediction_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "nlp",
            "lesson_46_toy_dialog_reopen_prediction",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.nlp.lesson_46_toy_dialog_reopen_prediction.train" in proc.stdout


def test_run_lesson_dry_run_resolves_nlp_dialog_resolution_owner_prediction_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "nlp",
            "lesson_47_toy_dialog_resolution_owner_prediction",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.nlp.lesson_47_toy_dialog_resolution_owner_prediction.train" in proc.stdout


def test_run_lesson_dry_run_resolves_nlp_dialog_resolution_action_prediction_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "nlp",
            "lesson_48_toy_dialog_resolution_action_prediction",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.nlp.lesson_48_toy_dialog_resolution_action_prediction.train" in proc.stdout


def test_run_lesson_dry_run_resolves_nlp_dialog_owner_handoff_prediction_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "nlp",
            "lesson_49_toy_dialog_owner_handoff_prediction",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.nlp.lesson_49_toy_dialog_owner_handoff_prediction.train" in proc.stdout
