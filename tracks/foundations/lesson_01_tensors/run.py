import argparse

import torch


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Demonstrate core PyTorch tensor shapes and operations."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    if __package__ is None:
        raise RuntimeError(
            "Run from repo root as module: python -m tracks.foundations.lesson_01_tensors.run"
        )
    parse_args(argv)

    x = torch.randn(2, 3, dtype=torch.float32)
    y = torch.randn(3, dtype=torch.float32)

    print("x.shape:", tuple(x.shape))
    print("y.shape:", tuple(y.shape))

    z = x + y  # broadcasting
    print("z.shape (x + y broadcast):", tuple(z.shape))

    a = torch.randn(4, 5)
    b = torch.randn(5, 6)
    c = a @ b
    print("matmul:", tuple(a.shape), "@", tuple(b.shape), "->", tuple(c.shape))

    logits = torch.randn(8, 10)
    probs = torch.softmax(logits, dim=1)
    print("softmax probs sum (first row):", float(probs[0].sum().item()))

    print("lesson_01_tensors: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
