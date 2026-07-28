"""Tests for scripts/run_lesson.py: llm track lessons."""

import subprocess
import sys

from _run_lesson_helpers import _repo_root


def test_run_lesson_dry_run_resolves_llm_mamba_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "llm",
            "lesson_03_compact_mamba_language_model",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.llm.lesson_03_compact_mamba_language_model.train" in proc.stdout


def test_run_lesson_dry_run_resolves_llm_instruction_tuning_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "llm",
            "lesson_04_compact_instruction_tuning",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.llm.lesson_04_compact_instruction_tuning.train" in proc.stdout


def test_run_lesson_dry_run_resolves_llm_prefix_tuning_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "llm",
            "lesson_05_compact_prefix_tuning",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.llm.lesson_05_compact_prefix_tuning.train" in proc.stdout


def test_run_lesson_dry_run_resolves_llm_preference_optimization_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "llm",
            "lesson_06_compact_preference_optimization",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.llm.lesson_06_compact_preference_optimization.train" in proc.stdout


def test_run_lesson_dry_run_resolves_llm_reward_modeling_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "llm",
            "lesson_07_compact_reward_modeling",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.llm.lesson_07_compact_reward_modeling.train" in proc.stdout


def test_run_lesson_dry_run_resolves_llm_span_corruption_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "llm",
            "lesson_08_compact_span_corruption",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.llm.lesson_08_compact_span_corruption.train" in proc.stdout


def test_run_lesson_dry_run_resolves_llm_rlhf_ppo_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "llm",
            "lesson_09_compact_rlhf_ppo",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.llm.lesson_09_compact_rlhf_ppo.train" in proc.stdout


def test_run_lesson_dry_run_resolves_llm_grpo_alignment_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "llm",
            "lesson_10_compact_grpo_alignment",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.llm.lesson_10_compact_grpo_alignment.train" in proc.stdout


def test_run_lesson_dry_run_resolves_llm_rag_language_model_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "llm",
            "lesson_11_compact_rag_language_model",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.llm.lesson_11_compact_rag_language_model.train" in proc.stdout


def test_run_lesson_dry_run_resolves_llm_transformer_interpretability_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "llm",
            "lesson_12_compact_transformer_interpretability",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.llm.lesson_12_compact_transformer_interpretability.train" in proc.stdout


def test_run_lesson_dry_run_resolves_llm_tool_calling_agent_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "llm",
            "lesson_13_compact_tool_calling_agent",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.llm.lesson_13_compact_tool_calling_agent.train" in proc.stdout


def test_run_lesson_dry_run_resolves_llm_replaced_token_detection_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "llm",
            "lesson_14_compact_replaced_token_detection_transformer",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.llm.lesson_14_compact_replaced_token_detection_transformer.train" in proc.stdout


def test_run_lesson_dry_run_resolves_llm_judge_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "llm",
            "lesson_15_compact_llm_judge",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.llm.lesson_15_compact_llm_judge.train" in proc.stdout


def test_run_lesson_dry_run_resolves_llm_multi_turn_memory_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "llm",
            "lesson_16_compact_multi_turn_memory_sft",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.llm.lesson_16_compact_multi_turn_memory_sft.train" in proc.stdout


def test_run_lesson_dry_run_resolves_llm_self_refine_prompting_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "llm",
            "lesson_17_compact_self_refine_prompting",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.llm.lesson_17_compact_self_refine_prompting.train" in proc.stdout


def test_run_lesson_dry_run_resolves_llm_reflection_memory_agent_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "llm",
            "lesson_18_compact_reflection_memory_agent",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.llm.lesson_18_compact_reflection_memory_agent.train" in proc.stdout


def test_run_lesson_dry_run_resolves_llm_plan_execute_prompting_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "llm",
            "lesson_19_compact_plan_execute_prompting",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.llm.lesson_19_compact_plan_execute_prompting.train" in proc.stdout


def test_run_lesson_dry_run_resolves_llm_citation_grounded_prompting_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "llm",
            "lesson_30_compact_citation_grounded_prompting",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.llm.lesson_30_compact_citation_grounded_prompting.train" in proc.stdout


def test_run_lesson_dry_run_resolves_llm_schema_constrained_prompting_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "llm",
            "lesson_31_compact_schema_constrained_prompting",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.llm.lesson_31_compact_schema_constrained_prompting.train" in proc.stdout


def test_run_lesson_dry_run_resolves_llm_json_constrained_prompting_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "llm",
            "lesson_32_compact_json_constrained_prompting",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.llm.lesson_32_compact_json_constrained_prompting.train" in proc.stdout


def test_run_lesson_dry_run_resolves_llm_function_signature_prompting_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "llm",
            "lesson_33_compact_function_signature_prompting",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.llm.lesson_33_compact_function_signature_prompting.train" in proc.stdout


def test_run_lesson_dry_run_resolves_llm_xml_constrained_prompting_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "llm",
            "lesson_34_compact_xml_constrained_prompting",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.llm.lesson_34_compact_xml_constrained_prompting.train" in proc.stdout


def test_run_lesson_dry_run_resolves_llm_regex_constrained_prompting_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "llm",
            "lesson_35_compact_regex_constrained_prompting",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.llm.lesson_35_compact_regex_constrained_prompting.train" in proc.stdout


def test_run_lesson_dry_run_resolves_llm_ebnf_constrained_prompting_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "llm",
            "lesson_36_compact_ebnf_constrained_prompting",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.llm.lesson_36_compact_ebnf_constrained_prompting.train" in proc.stdout


def test_run_lesson_dry_run_resolves_llm_sql_constrained_prompting_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "llm",
            "lesson_37_compact_sql_constrained_prompting",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.llm.lesson_37_compact_sql_constrained_prompting.train" in proc.stdout


def test_run_lesson_dry_run_resolves_llm_yaml_constrained_prompting_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "llm",
            "lesson_38_compact_yaml_constrained_prompting",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.llm.lesson_38_compact_yaml_constrained_prompting.train" in proc.stdout


def test_run_lesson_dry_run_resolves_llm_csv_constrained_prompting_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "llm",
            "lesson_39_compact_csv_constrained_prompting",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.llm.lesson_39_compact_csv_constrained_prompting.train" in proc.stdout


def test_run_lesson_dry_run_resolves_llm_toml_constrained_prompting_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "llm",
            "lesson_40_compact_toml_constrained_prompting",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.llm.lesson_40_compact_toml_constrained_prompting.train" in proc.stdout


def test_run_lesson_dry_run_resolves_llm_markdown_table_constrained_prompting_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "llm",
            "lesson_41_compact_markdown_table_constrained_prompting",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.llm.lesson_41_compact_markdown_table_constrained_prompting.train" in proc.stdout


def test_run_lesson_dry_run_resolves_llm_ini_constrained_prompting_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "llm",
            "lesson_42_compact_ini_constrained_prompting",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.llm.lesson_42_compact_ini_constrained_prompting.train" in proc.stdout


def test_run_lesson_dry_run_resolves_llm_tsv_constrained_prompting_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "llm",
            "lesson_43_compact_tsv_constrained_prompting",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.llm.lesson_43_compact_tsv_constrained_prompting.train" in proc.stdout
