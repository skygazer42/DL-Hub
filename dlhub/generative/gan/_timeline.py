"""GAN timeline metadata (best effort, for docs and CLI)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TimelineEntry:
    year: int | None
    family: str
    method: str
    group: str
    reference: str | None = None


_ENTRIES: list[TimelineEntry] = [
    TimelineEntry(2015, "dcgan", "DCGAN (deep convolutional GAN baseline)", "vanilla_adversarial"),
    TimelineEntry(2016, "lsgan", "LSGAN (least-squares adversarial loss)", "vanilla_adversarial"),
    TimelineEntry(2017, "wgan", "WGAN (Wasserstein GAN objective)", "vanilla_adversarial"),
    TimelineEntry(2017, "wgangp", "WGAN-GP (gradient penalty WGAN)", "vanilla_adversarial"),
    TimelineEntry(
        2018, "hingegan", "HingeGAN (hinge adversarial objective)", "vanilla_adversarial"
    ),
    TimelineEntry(
        2019,
        "relativistic_gan",
        "Relativistic GAN (relative realism discriminator)",
        "vanilla_adversarial",
    ),
    TimelineEntry(2018, "dragan", "DRAGAN (local gradient regularized GAN)", "vanilla_adversarial"),
    TimelineEntry(
        2014, "cgan", "Conditional GAN (label-conditioned generation)", "conditional_gan"
    ),
    TimelineEntry(2017, "acgan", "ACGAN (auxiliary classifier GAN)", "conditional_gan"),
    TimelineEntry(
        2018,
        "projection_gan",
        "Projection GAN (projection discriminator conditioning)",
        "conditional_gan",
    ),
    TimelineEntry(2016, "infogan", "InfoGAN (mutual information guided GAN)", "conditional_gan"),
    TimelineEntry(2016, "stackgan", "StackGAN (stacked text-conditioned GAN)", "conditional_gan"),
    TimelineEntry(2017, "pix2pix", "Pix2Pix (paired image translation GAN)", "image_translation"),
    TimelineEntry(2017, "cyclegan", "CycleGAN (unpaired translation GAN)", "image_translation"),
    TimelineEntry(2017, "dualgan", "DualGAN (dual learning translation GAN)", "image_translation"),
    TimelineEntry(2017, "unit", "UNIT (shared latent space translation GAN)", "image_translation"),
    TimelineEntry(
        2020, "cutgan", "CUTGAN (contrastive unpaired translation GAN)", "image_translation"
    ),
    TimelineEntry(2019, "stylegan", "StyleGAN (style-based generator)", "high_fidelity"),
    TimelineEntry(2020, "stylegan2", "StyleGAN2 (improved style synthesis)", "high_fidelity"),
    TimelineEntry(2021, "stylegan3", "StyleGAN3 (alias-free style synthesis)", "high_fidelity"),
    TimelineEntry(2018, "biggan", "BigGAN (large-scale class-conditional GAN)", "high_fidelity"),
    TimelineEntry(2019, "sagan", "SAGAN (self-attention GAN)", "high_fidelity"),
    TimelineEntry(2017, "progan", "ProGAN (progressive growing GAN)", "high_fidelity"),
    TimelineEntry(2021, "transgan", "TransGAN (transformer-only GAN)", "high_fidelity"),
]


def entries() -> list[TimelineEntry]:
    return list(_ENTRIES)


def by_family() -> dict[str, TimelineEntry]:
    return {e.family: e for e in _ENTRIES}
