from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    image_size: int = 28
    in_channels: int = 1
    latent_channels: int = 4
    latent_size: int = 7
    hidden_channels: int = 16
    num_diffusion_steps: int = 8
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    recon_weight: float = 0.25


class LatentDenoiser(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.time_embed = nn.Embedding(cfg.num_diffusion_steps, cfg.hidden_channels)
        self.in_proj = nn.Conv2d(cfg.latent_channels, cfg.hidden_channels, kernel_size=3, padding=1)
        self.net = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(cfg.hidden_channels, cfg.hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(cfg.hidden_channels, cfg.hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(cfg.hidden_channels, cfg.latent_channels, kernel_size=3, padding=1),
        )

    def forward(self, latents: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(latents)
        t = self.time_embed(timesteps).unsqueeze(-1).unsqueeze(-1)
        return self.net(h + t)


class LatentDiffusionModel(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        if cfg.image_size != 28 or cfg.latent_size != 7:
            raise ValueError("This toy lesson currently supports 28x28 images and 7x7 latents only.")

        self.encoder = nn.Sequential(
            nn.Conv2d(cfg.in_channels, cfg.hidden_channels, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(
                cfg.hidden_channels,
                cfg.hidden_channels,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.SiLU(),
            nn.Conv2d(cfg.hidden_channels, cfg.latent_channels, kernel_size=3, padding=1),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                cfg.latent_channels,
                cfg.hidden_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.SiLU(),
            nn.ConvTranspose2d(
                cfg.hidden_channels,
                cfg.hidden_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.SiLU(),
            nn.Conv2d(cfg.hidden_channels, cfg.in_channels, kernel_size=3, padding=1),
        )
        self.denoiser = LatentDenoiser(cfg)

        betas = torch.linspace(cfg.beta_start, cfg.beta_end, cfg.num_diffusion_steps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("sqrt_alpha_bars", torch.sqrt(alpha_bars))
        self.register_buffer("sqrt_one_minus_alpha_bars", torch.sqrt(1.0 - alpha_bars))

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return self.encoder(images)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.decoder(latents))

    def add_noise(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(latents)
        view_shape = (-1, 1, 1, 1)
        sqrt_alpha_bar = self.sqrt_alpha_bars[timesteps].view(view_shape)
        sqrt_one_minus = self.sqrt_one_minus_alpha_bars[timesteps].view(view_shape)
        noisy_latents = sqrt_alpha_bar * latents + sqrt_one_minus * noise
        return noisy_latents, noise

    def predict_noise(self, noisy_latents: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        return self.denoiser(noisy_latents, timesteps)

    def sample(self, batch_size: int, device: torch.device | str) -> torch.Tensor:
        device = torch.device(device)
        latents = torch.randn(
            (batch_size, self.cfg.latent_channels, self.cfg.latent_size, self.cfg.latent_size),
            device=device,
        )
        for step in reversed(range(self.cfg.num_diffusion_steps)):
            t = torch.full((batch_size,), step, device=device, dtype=torch.long)
            noise_pred = self.predict_noise(latents, t)
            alpha = self.alphas[t].view(-1, 1, 1, 1)
            alpha_bar = self.alpha_bars[t].view(-1, 1, 1, 1)
            beta = self.betas[t].view(-1, 1, 1, 1)

            latents = (latents - beta / torch.sqrt(1.0 - alpha_bar) * noise_pred) / torch.sqrt(alpha)
            if step > 0:
                latents = latents + torch.sqrt(beta) * torch.randn_like(latents)

        return self.decode(latents)


def diffusion_loss(
    *,
    noise_pred: torch.Tensor,
    noise: torch.Tensor,
    recon_images: torch.Tensor,
    target_images: torch.Tensor,
    recon_weight: float = 0.25,
) -> torch.Tensor:
    noise_loss = torch.mean((noise_pred - noise) ** 2)
    recon_loss = torch.mean((recon_images - target_images) ** 2)
    return noise_loss + float(recon_weight) * recon_loss

