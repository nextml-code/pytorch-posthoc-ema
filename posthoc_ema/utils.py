"""Common utility functions for EMA implementations."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


def exists(val):
    """Check if a value exists (is not None)."""
    return val is not None


def sigma_rel_to_gamma(sigma_rel: float) -> float:
    """
    Convert relative standard deviation (σrel) to gamma parameter.

    Args:
        sigma_rel: Relative standard deviation (e.g., 0.10 for 10% EMA length)

    Returns:
        float: Corresponding gamma value
    """
    t = sigma_rel**-2
    return np.roots([1, 7, 16 - t, 12 - t]).real.max().item()


def p_dot_p(t_a: Tensor, gamma_a: Tensor, t_b: Tensor, gamma_b: Tensor) -> Tensor:
    """
    Compute dot product between two power function EMA profiles.

    Args:
        t_a: First timestep tensor
        gamma_a: First gamma parameter tensor
        t_b: Second timestep tensor
        gamma_b: Second gamma parameter tensor

    Returns:
        Tensor: Dot product between the profiles
    """
    t_ratio = t_a / t_b
    t_exp = torch.where(t_a < t_b, gamma_b, -gamma_a)
    t_max = torch.maximum(t_a, t_b)
    num = (gamma_a + 1) * (gamma_b + 1) * t_ratio**t_exp
    den = (gamma_a + gamma_b + 1) * t_max
    return num / den


def solve_weights(t_i: Tensor, gamma_i: Tensor, t_r: Tensor, gamma_r: Tensor) -> Tensor:
    """
    Solve for optimal weights to synthesize target EMA profile.

    Implements Algorithm 3 from the paper.

    Args:
        t_i: Timesteps of stored checkpoints
        gamma_i: Gamma values of stored checkpoints
        t_r: Target timestep
        gamma_r: Target gamma value

    Returns:
        Tensor: Optimal weights for combining checkpoints
    """
    rv = lambda x: x.double().reshape(-1, 1)
    cv = lambda x: x.double().reshape(1, -1)
    A = p_dot_p(rv(t_i), rv(gamma_i), cv(t_i), cv(gamma_i))
    b = p_dot_p(rv(t_i), rv(gamma_i), cv(t_r), cv(gamma_r))
    return torch.linalg.solve(A, b) 