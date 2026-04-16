import json
import os

import pytest

torch = pytest.importorskip("torch")


def test_sentence_denoising_batch_contract() -> None:
    from tracks.nlp.lesson_23_toy_sentence_denoising_autoencoder.data import (
        DataConfig,
        get_dataloaders,
    )
    from tracks.nlp.lesson_23_toy_sentence_denoising_autoencoder.model import (
        DenoisingSeq2Seq,
        ModelConfig,
        reconstruction_token_accuracy,
    )

    train_loader, _, vocab = get_dataloaders(
        DataConfig(
            num_samples=48,
            batch_size=6,
            max_length=12,
            corruption_prob=0.35,
            val_fraction=0.25,
            seed=0,
            num_workers=0,
        )
    )
    inputs, targets = next(iter(train_loader))

    assert tuple(inputs["src_ids"].shape) == (6, 12)
    assert tuple(inputs["src_mask"].shape) == (6, 12)
    assert tuple(inputs["tgt_in_ids"].shape) == (6, 13)
    assert tuple(targets["tgt_out_ids"].shape) == (6, 13)

    model = DenoisingSeq2Seq(
        ModelConfig(
            vocab_size=vocab.size,
            pad_id=vocab.pad_id,
            bos_id=vocab.bos_id,
            eos_id=vocab.eos_id,
            embed_dim=32,
            hidden_dim=48,
            dropout=0.0,
        )
    )
    outputs = model(
        src_ids=inputs["src_ids"],
        src_mask=inputs["src_mask"],
        tgt_in_ids=inputs["tgt_in_ids"],
    )
    assert tuple(outputs["logits"].shape) == (6, 13, vocab.size)

    loss = torch.nn.CrossEntropyLoss(ignore_index=vocab.pad_id)(
        outputs["logits"].reshape(-1, vocab.size),
        targets["tgt_out_ids"].reshape(-1),
    )
    assert torch.isfinite(loss)
    assert 0.0 <= reconstruction_token_accuracy(outputs["logits"], targets["tgt_out_ids"], vocab.pad_id) <= 1.0


def test_sentence_denoising_training_smoke(tmp_path) -> None:
    from tracks.nlp.lesson_23_toy_sentence_denoising_autoencoder.data import DataConfig
    from tracks.nlp.lesson_23_toy_sentence_denoising_autoencoder.train import (
        TrainConfig,
        run_training,
    )

    os.environ["DLHUB_OUTPUTS_DIR"] = str(tmp_path / "outputs")
    try:
        exit_code = run_training(
            TrainConfig(
                epochs=1,
                learning_rate=2e-3,
                seed=42,
                device="cpu",
                max_train_batches=2,
                max_eval_batches=1,
                run_name="pytest_sentence_denoising_smoke",
                embed_dim=32,
                hidden_dim=48,
                dropout=0.0,
            ),
            DataConfig(
                num_samples=64,
                batch_size=8,
                max_length=12,
                corruption_prob=0.35,
                val_fraction=0.25,
                seed=4,
                num_workers=0,
            ),
        )
        assert exit_code == 0
    finally:
        os.environ.pop("DLHUB_OUTPUTS_DIR", None)

    run_dir = (
        tmp_path
        / "outputs"
        / "nlp"
        / "lesson_23_toy_sentence_denoising_autoencoder"
        / "pytest_sentence_denoising_smoke"
    )
    assert (run_dir / "config.json").is_file()
    assert (run_dir / "vocab.json").is_file()
    assert (run_dir / "metrics.jsonl").is_file()
    assert (run_dir / "samples.jsonl").is_file()
    assert (run_dir / "checkpoints" / "checkpoint.pt").is_file()

    metric_row = json.loads((run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert metric_row["epoch"] == 1
    assert metric_row["train_loss"] >= 0.0
    assert 0.0 <= metric_row["train_token_acc"] <= 1.0
    assert metric_row["eval_loss"] >= 0.0
    assert 0.0 <= metric_row["eval_token_acc"] <= 1.0
