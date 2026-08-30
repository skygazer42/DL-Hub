import random

import numpy as np


def set_seed(
    seed: int,
    *,
    deterministic: bool = False,
    warn_only: bool = False,
) -> None:
    """Seed Python, NumPy, and Torch RNGs when Torch is importable.

    Torch seeding is done lazily to avoid forcing a torch dependency for
    non-torch-only parts of the repo. With ``deterministic=True``, PyTorch's
    process-wide deterministic-algorithm mode is enabled and cuDNN autotuning
    is disabled. The default leaves any existing deterministic settings alone.

    Import failures remain optional; once Torch imports successfully, seeding
    and backend-configuration errors are deliberately propagated.
    """

    if warn_only and not deterministic:
        raise ValueError("warn_only requires deterministic=True")

    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except (ImportError, OSError, RuntimeError):
        # Keep the utility usable when Torch is absent or cannot be imported.
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=bool(warn_only))
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
