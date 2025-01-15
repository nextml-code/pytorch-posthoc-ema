"""Common utility functions for EMA implementations."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


def exists(val):
    """Check if a value exists (is not None)."""
    return val is not None


def beta_to_sigma_rel(beta: float) -> float:
    """
    Convert EMA decay rate (β) to relative standard deviation (σrel).

    Args:
        beta: EMA decay rate (e.g., 0.9999 for strong smoothing)

    Returns:
        float: Corresponding relative standard deviation
    """
    if not 0 < beta < 1:
        raise ValueError(f"Beta must be between 0 and 1, got {beta}")
    # From β = 1 - 1/γ, we get γ = 1/(1-β)
    gamma = 1 / (1 - beta)
    # Then use gamma_to_sigma_rel formula from paper
    return float(np.sqrt((gamma + 1) / ((gamma + 2) * (gamma + 3))))


def sigma_rel_to_beta(sigma_rel: float) -> float:
    """
    Convert relative standard deviation (σrel) to EMA decay rate (β).

    Args:
        sigma_rel: Relative standard deviation (e.g., 0.10 for 10% EMA length)

    Returns:
        float: Corresponding beta value
    """
    if sigma_rel <= 0:
        raise ValueError(f"sigma_rel must be positive, got {sigma_rel}")
    gamma = sigma_rel_to_gamma(sigma_rel)
    # From γ = 1/(1-β), we get β = 1 - 1/(γ+1)
    return float(1 - 1 / (gamma + 1))


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
    # Handle t=0 case: if both times are 0, ratio is 1
    t_ratio = torch.where(
        (t_a == 0) & (t_b == 0),
        torch.ones_like(t_a),
        t_a / torch.where(t_b == 0, torch.ones_like(t_b), t_b)
    )
    
    t_exp = torch.where(t_a < t_b, gamma_b, -gamma_a)
    t_max = torch.maximum(t_a, t_b)
    
    # Handle t=0 case: if both times are 0, max is 1
    t_max = torch.where(
        (t_a == 0) & (t_b == 0),
        torch.ones_like(t_max),
        t_max
    )
    
    # Print debug info for first few values
    if t_a.shape[0] < 10:  # Only print for small tensors
        print(f"\nt_ratio shape: {t_ratio.shape}")
        print(f"t_ratio first few: {t_ratio[:5, :5]}")
        print(f"t_exp first few: {t_exp[:5, :5]}")
        print(f"t_max first few: {t_max[:5, :5]}")
    
    num = (gamma_a + 1) * (gamma_b + 1) * t_ratio**t_exp
    den = (gamma_a + gamma_b + 1) * t_max
    
    if t_a.shape[0] < 10:  # Only print for small tensors
        print(f"num first few: {num[:5, :5]}")
        print(f"den first few: {den[:5, :5]}")
        print(f"result first few: {(num/den)[:5, :5]}")
    
    return num / den


def solve_weights(t_i: Tensor, gamma_i: Tensor, t_r: Tensor, gamma_r: Tensor) -> Tensor:
    """
    Solve for optimal weights to synthesize target EMA profile.

    Args:
        t_i: Timesteps for source profiles
        gamma_i: Gamma values for source profiles
        t_r: Target timesteps
        gamma_r: Target gamma value

    Returns:
        Tensor: Optimal weights for combining source profiles
    """
    # Reshape tensors for matrix operations
    rv = lambda x: x.reshape(-1, 1)  # Column vector
    cv = lambda x: x.reshape(1, -1)  # Row vector
    
    # Compute matrices A and b using p_dot_p
    A = p_dot_p(rv(t_i), rv(gamma_i), cv(t_i), cv(gamma_i))
    b = p_dot_p(rv(t_i), rv(gamma_i), cv(t_r), cv(gamma_r))
    
    # Solve linear system
    return torch.linalg.solve(A, b) 