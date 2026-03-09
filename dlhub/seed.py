
import random

import numpy as np


def set_seed(seed: int) -> None:
    """Seed Python and NumPy RNGs.

    Torch seeding is done lazily to avoid forcing a torch dependency for
    non-torch-only parts of the repo.
    """

    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        # Keep seed utility usable even when torch isn't installed.
        pass
