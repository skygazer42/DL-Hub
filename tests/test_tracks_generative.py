import pytest


torch = pytest.importorskip("torch")


def test_generative_vae_forward_shapes_smoke() -> None:
    from tracks.generative.lesson_01_vae_mnist.model import ModelConfig, VAE, vae_loss

    cfg = ModelConfig(latent_dim=8, hidden_dim=64)
    model = VAE(cfg)
    x = torch.rand((4, 28 * 28), dtype=torch.float32)

    recon_logits, mu, logvar = model(x)
    assert recon_logits.shape == x.shape
    assert mu.shape == (4, 8)
    assert logvar.shape == (4, 8)

    loss, recon, kl = vae_loss(recon_logits=recon_logits, x=x, mu=mu, logvar=logvar, beta=1.0)
    assert torch.isfinite(loss)
    assert torch.isfinite(recon)
    assert torch.isfinite(kl)


def test_generative_vae_fake_dataloaders_smoke() -> None:
    from tracks.generative.lesson_01_vae_mnist.data import DataConfig, get_dataloaders

    train_loader, val_loader = get_dataloaders(
        DataConfig(dataset="fake", num_samples=64, batch_size=8, seed=0, num_workers=0, val_fraction=0.2)
    )
    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))
    assert train_batch.shape == (8, 1, 28, 28)
    assert val_batch.shape[1:] == (1, 28, 28)


def test_generative_gan_forward_shapes_smoke() -> None:
    from tracks.generative.lesson_02_gan_mnist.model import GAN, ModelConfig

    cfg = ModelConfig(z_dim=16, hidden_dim=64)
    bundle = GAN(cfg)

    z = torch.randn((4, cfg.z_dim), dtype=torch.float32)
    fake = bundle.generator(z)
    assert fake.shape == (4, 1, 28, 28)

    logits = bundle.discriminator(fake)
    assert logits.shape == (4,)

