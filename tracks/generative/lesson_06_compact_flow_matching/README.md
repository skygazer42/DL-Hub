# Lesson 06: Compact Flow Matching

This lesson introduces the simplest continuous-time flow-matching recipe in a CPU-friendly form:

- draw a synthetic grayscale image `x_data`
- draw Gaussian noise `x_noise`
- linearly interpolate to an intermediate state `x_t = (1 - t) x_noise + t x_data`
- train a small convolutional network to predict the constant transport velocity `x_data - x_noise`

Sampling starts from random noise and integrates the learned velocity field with a short Euler solver.
That makes the lesson a clean neighbor-fill continuation after DDPM and latent diffusion without
relying on a discrete reverse-diffusion schedule.
