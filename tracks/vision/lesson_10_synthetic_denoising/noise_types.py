
# Kept in a tiny module so CLI "list" commands can stay fast and avoid importing torch.
SUPPORTED_NOISE_TYPES: tuple[str, ...] = (
    "gaussian",
    "gaussian_var",
    "gaussian_impulse",
    "poisson",
    "poisson_gaussian",
    "impulse",
    "clustered_impulse",
    "shot_read",
    "speckle",
    "speckle_read",
    "stripe",
    "rain",
    "block_bias",
    "correlated_gaussian",
    "colored_gaussian",
    "quantization",
    "dead_hot",
    "line_defect",
    "rowcol_bias",
    "mixed",
)


def list_supported_noise_types() -> list[str]:
    # Canonical names used in docs/CLI. The dataset also accepts a few aliases (see _make_noisy).
    return list(SUPPORTED_NOISE_TYPES)
