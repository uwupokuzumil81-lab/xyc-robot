# Copyright 2025 The HuggingFace Inc. team.
# SPDX-License-Identifier: Apache-2.0
"""π0.5 RTC's linear-prefix guidance, adapted from LeRobot v0.5.1."""

import torch


MAX_GUIDANCE_WEIGHT = 10.0


def guided_denoise_step(x_t, previous, inference_delay, execution_horizon, time, denoise):
    """Guide the denoising velocity toward the remaining normalized action prefix."""
    _, steps, action_dim = x_t.shape
    prefix_length, prefix_dim = previous.shape[1:]
    end = min(execution_horizon, prefix_length, steps)
    start = min(inference_delay, end)
    linear = torch.linspace(1, 0, end - start + 2)[1:-1]
    weights = torch.cat((torch.ones(start), linear, torch.zeros(steps - end)))
    weights = weights.to(x_t.device).view(1, steps, 1)

    # The model samples in padded action space; XYC retains only the actual axes.
    previous_padded = torch.zeros_like(x_t)
    previous_padded[:, :prefix_length, :prefix_dim] = previous
    x_t = x_t.detach().clone()

    with torch.enable_grad():
        velocity = denoise(x_t)
        # LeRobot v0.5.1 treats the denoiser velocity as fixed for the correction.
        x_t.requires_grad_(True)
        estimate = x_t - time * velocity
        error = ((previous_padded - estimate) * weights).detach()
        correction = torch.autograd.grad(estimate, x_t, error)[0]

    tau = torch.as_tensor(1 - time)
    remaining_squared = (1 - tau).square()
    limit = torch.as_tensor(MAX_GUIDANCE_WEIGHT)
    coefficient = torch.nan_to_num((1 - tau) / tau, posinf=limit)
    inverse_r2 = (remaining_squared + tau.square()) / remaining_squared
    weight = torch.nan_to_num(coefficient * inverse_r2, posinf=limit)
    return velocity - torch.minimum(weight, limit) * correction
