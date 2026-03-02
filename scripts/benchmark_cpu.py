from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def _ensure_repo_root_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DL-Hub: tiny CPU benchmark for the training loop.")
    parser.add_argument("--batches", type=int, default=100, help="Number of train batches to run.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-samples", type=int, default=2048)
    return parser.parse_args()


def main() -> int:
    _ensure_repo_root_on_path()
    args = parse_args()

    try:
        import torch
    except Exception as exc:
        print("benchmark_cpu: torch not available; skipping.")
        print(f"- reason: {exc}")
        return 0

    from dlhub.data.toy import ToyClassificationConfig, make_toy_classification_dataloaders
    from dlhub.seed import set_seed
    from dlhub.training.loop import fit_classifier

    set_seed(0)
    train_loader, _ = make_toy_classification_dataloaders(
        ToyClassificationConfig(
            num_samples=int(args.num_samples),
            num_features=32,
            noise_std=0.1,
            val_fraction=0.2,
            seed=0,
        ),
        batch_size=int(args.batch_size),
        num_workers=0,
    )

    model = torch.nn.Sequential(
        torch.nn.Linear(32, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 2),
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    criterion = torch.nn.CrossEntropyLoss()
    device = torch.device("cpu")
    model.to(device)

    # Warm-up a couple of batches to reduce first-iteration effects.
    fit_classifier(
        model=model,
        loader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        max_batches=2,
    )

    t0 = time.perf_counter()
    fit_classifier(
        model=model,
        loader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        max_batches=int(args.batches),
    )
    t1 = time.perf_counter()

    elapsed = t1 - t0
    batches_per_sec = (args.batches / elapsed) if elapsed > 0 else float("inf")
    print("benchmark_cpu: OK")
    print(f"- batches: {args.batches}")
    print(f"- elapsed_sec: {elapsed:.4f}")
    print(f"- batches_per_sec: {batches_per_sec:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

