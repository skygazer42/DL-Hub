
import argparse
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class NewLesson:
    track: str
    lesson: str
    title: str


def parse_args() -> NewLesson:
    parser = argparse.ArgumentParser(
        description="Create a new lesson skeleton under tracks/<track>/<lesson>/"
    )
    parser.add_argument("--track", required=True, help="e.g. vision | nlp | gnn | foundations")
    parser.add_argument("--lesson", required=True, help="e.g. lesson_02_mnist_mlp")
    parser.add_argument(
        "--title", default="", help="Optional title for README (defaults to lesson name)"
    )
    args = parser.parse_args()

    title = args.title.strip() or args.lesson
    return NewLesson(track=args.track.strip(), lesson=args.lesson.strip(), title=title)


def write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> int:
    spec = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    lesson_dir = repo_root / "tracks" / spec.track / spec.lesson

    if lesson_dir.exists():
        raise FileExistsError(f"Lesson directory already exists: {lesson_dir}")

    write_file(lesson_dir / "__init__.py", f'"""Lesson: {spec.lesson}."""\n')
    write_file(
        lesson_dir / "README.md",
        f"# {spec.title}\n\n"
        "## Goal\n\n- TODO\n\n"
        "## Run (smoke)\n\n"
        "```bash\n"
        f"python -m tracks.{spec.track}.{spec.lesson}.train --dataset fake --epochs 1 --max-train-batches 2 --max-eval-batches 2\n"
        "```\n",
    )
    write_file(
        lesson_dir / "model.py",
        "import torch\n"
        "import torch.nn as nn\n\n\n"
        "class Model(nn.Module):\n"
        "    def __init__(self) -> None:\n"
        "        super().__init__()\n"
        "        self.net = nn.Identity()\n\n"
        "    def forward(self, x: torch.Tensor) -> torch.Tensor:\n"
        "        return self.net(x)\n",
    )
    write_file(
        lesson_dir / "data.py",
        "from dataclasses import dataclass\n\n"
        "from torch.utils.data import DataLoader\n\n\n"
        "@dataclass(frozen=True)\n"
        "class DataConfig:\n"
        '    dataset: str = "fake"\n'
        "    batch_size: int = 64\n"
        "    num_workers: int = 0\n\n\n"
        "def get_dataloaders(config: DataConfig) -> tuple[DataLoader, DataLoader]:\n"
        "    raise NotImplementedError\n",
    )
    write_file(
        lesson_dir / "train.py",
        "import argparse\n\n\n"
        "def main() -> int:\n"
        "    if __package__ is None:\n"
        "        raise RuntimeError(\n"
        '            "Run from repo root as module: python -m tracks.<track>.<lesson>.train"\n'
        "        )\n"
        "    parser = argparse.ArgumentParser()\n"
        "    parser.parse_args()\n"
        "    raise NotImplementedError\n\n\n"
        'if __name__ == "__main__":\n'
        "    raise SystemExit(main())\n",
    )

    print(f"Created: {lesson_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
