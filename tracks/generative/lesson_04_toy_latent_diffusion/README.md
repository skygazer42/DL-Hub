# Lesson 04: Toy Latent Diffusion

This lesson keeps latent diffusion intentionally small:

- encode a `28x28` grayscale toy image into a `7x7` latent grid
- add noise and train a denoiser in latent space
- decode denoised latents back into pixel space

The implementation is pure PyTorch and CPU friendly so it can be used as a smoke-testable introduction after the pixel-space diffusion lesson.

