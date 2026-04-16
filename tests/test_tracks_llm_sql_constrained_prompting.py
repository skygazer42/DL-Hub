import os

import pytest

torch = pytest.importorskip("torch")


def test_llm_sql_constrained_prompting_batch_mask_and_loss_smoke() -> None:
    from tracks.llm.lesson_37_toy_sql_constrained_prompting.data import DataConfig, get_dataloaders
    from tracks.llm.lesson_37_toy_sql_constrained_prompting.model import (
        ModelConfig,
        SqlConstrainedPromptingTransformerLM,
    )

    train_loader, _, vocab = get_dataloaders(
        DataConfig(
            num_samples=64,
            batch_size=8,
            seq_length=42,
            base_vocab_size=32,
            val_fraction=0.25,
            seed=0,
            num_workers=0,
        )
    )
    inputs, labels = next(iter(train_loader))
    assert tuple(inputs["input_ids"].shape) == (8, 42)
    assert tuple(inputs["attention_mask"].shape) == (8, 42)
    assert tuple(labels.shape) == (8, 42)

    sql_positions = (inputs["input_ids"] == int(vocab.sql_token_id)).to(torch.int64).argmax(dim=1)
    for row_idx, sql_pos in enumerate(sql_positions.tolist()):
        assert torch.all(labels[row_idx, : int(sql_pos)] == int(vocab.ignore_index))
    assert (labels != int(vocab.ignore_index)).any()

    model = SqlConstrainedPromptingTransformerLM(
        ModelConfig(
            vocab_size=vocab.size,
            pad_id=vocab.pad_id,
            max_length=42,
            embed_dim=48,
            num_heads=4,
            num_layers=2,
            ff_dim=96,
            dropout=0.0,
        )
    )
    logits = model(inputs)
    assert tuple(logits.shape) == (8, 42, vocab.size)

    loss = torch.nn.CrossEntropyLoss(ignore_index=int(vocab.ignore_index))(
        logits.reshape(-1, vocab.size),
        labels.reshape(-1),
    )
    assert torch.isfinite(loss)
    loss.backward()


def test_llm_sql_constrained_prompting_training_smoke(tmp_path) -> None:
    from tracks.llm.lesson_37_toy_sql_constrained_prompting.data import DataConfig
    from tracks.llm.lesson_37_toy_sql_constrained_prompting.train import TrainConfig, run_training

    os.environ["DLHUB_OUTPUTS_DIR"] = str(tmp_path / "outputs")
    try:
        exit_code = run_training(
            TrainConfig(
                epochs=1,
                learning_rate=2e-3,
                seed=37,
                device="cpu",
                max_train_batches=2,
                max_eval_batches=1,
                run_name="pytest_sql_constrained_prompting_smoke",
                generation_tokens=10,
                embed_dim=48,
                num_heads=4,
                num_layers=2,
                ff_dim=96,
                dropout=0.0,
            ),
            DataConfig(
                num_samples=80,
                batch_size=8,
                seq_length=42,
                base_vocab_size=32,
                val_fraction=0.25,
                seed=11,
                num_workers=0,
            ),
        )
        assert exit_code == 0
    finally:
        os.environ.pop("DLHUB_OUTPUTS_DIR", None)

    run_dir = (
        tmp_path
        / "outputs"
        / "llm"
        / "lesson_37_toy_sql_constrained_prompting"
        / "pytest_sql_constrained_prompting_smoke"
    )
    assert (run_dir / "config.json").is_file()
    assert (run_dir / "vocab.json").is_file()
    assert (run_dir / "metrics.jsonl").is_file()
    assert (run_dir / "samples.jsonl").is_file()
    assert (run_dir / "checkpoints" / "checkpoint.pt").is_file()
