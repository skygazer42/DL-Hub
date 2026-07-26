from __future__ import annotations

from torch import nn

from ._common import build_toy_doc_binarizer, smoke_test_doc_binarizer


_VARIANTS: dict[str, dict[str, int]] = {
    "prompt_docbin_tiny": {"width": 24, "depth": 1},
    "prompt_docbin_small": {"width": 36, "depth": 2},
    "prompt_docbin_base": {"width": 48, "depth": 3},
}


def build_prompt_docbin_doc_binarizer(
    *, in_channels: int, variant: str = "prompt_docbin_small", width_mult: float = 1.0
) -> nn.Module:
    return build_toy_doc_binarizer(
        family="prompt_docbin",
        mode="prompt",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_doc_binarizer(build_prompt_docbin_doc_binarizer, "prompt_docbin_tiny")
