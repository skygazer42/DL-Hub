import argparse
import sys
from collections.abc import Iterable
from pathlib import Path


def _ensure_repo_root_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))


def _summarize(obj) -> str:
    from dlhub.cli_utils import summarize_output

    return summarize_output(obj)


def _print_lines(lines: Iterable[str], *, limit: int = 80) -> None:
    from dlhub.cli_utils import print_limited

    print_limited(lines, limit=limit, annotate_fidelity=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NLP local model zoo utilities (no downloads).")

    parser.add_argument("--list", action="store_true", help="List available architecture ids.")
    parser.add_argument(
        "--search", type=str, default=None, help="Filter list by substring (case-insensitive)."
    )
    parser.add_argument("--limit", type=int, default=80, help="Max lines to print when listing.")

    parser.add_argument(
        "--smoke",
        type=str,
        default=None,
        metavar="ARCH_ID",
        help="Run a forward smoke on an arch id.",
    )
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size for smoke inputs.")
    parser.add_argument("--vocab-size", type=int, default=128, help="Vocab size for local models.")
    parser.add_argument("--pad-id", type=int, default=0, help="Padding token id.")
    parser.add_argument(
        "--max-length", type=int, default=32, help="Sequence length for smoke inputs."
    )
    parser.add_argument("--num-classes", type=int, default=4, help="Classifier output classes.")
    parser.add_argument(
        "--width-mult",
        type=float,
        default=1.0,
        help="Width multiplier for local models that support it.",
    )
    parser.add_argument(
        "--dropout", type=float, default=0.1, help="Dropout for local models that support it."
    )

    return parser.parse_args()


def main() -> int:
    _ensure_repo_root_on_path()

    from dlhub.nlp.local_zoo import build_local_model, list_local_arches

    args = parse_args()

    if not args.list and args.smoke is None:
        print("Nothing to do. Try one of:")
        print("- python scripts/nlp_zoo.py --list")
        print("- python scripts/nlp_zoo.py --smoke nl:bert_tiny")
        return 2

    arches = list_local_arches()
    if args.search:
        needle = str(args.search).lower()
        arches = [a for a in arches if needle in a.lower()]

    if args.list:
        print("NLP local zoo")
        print(f"- total_arches={len(arches)}")
        print("")
        _print_lines(arches, limit=int(args.limit))

    if args.smoke is not None:
        arch_id = str(args.smoke).strip()
        if ":" not in arch_id:
            arch_id = f"nl:{arch_id}"

        import torch

        b = int(args.batch_size)
        t = int(args.max_length)

        input_ids = torch.zeros((b, t), dtype=torch.long)
        attention_mask = torch.zeros((b, t), dtype=torch.float32)
        attention_mask[:, : max(1, t // 2)] = 1.0

        model = build_local_model(
            arch_id,
            vocab_size=int(args.vocab_size),
            pad_id=int(args.pad_id),
            max_length=t,
            num_classes=int(args.num_classes),
            width_mult=float(args.width_mult),
            dropout=float(args.dropout),
        )
        model.eval()
        with torch.no_grad():
            out = model({"input_ids": input_ids, "attention_mask": attention_mask})

        print("")
        print(f"smoke: {arch_id}")
        print(f"- output={_summarize(out)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
