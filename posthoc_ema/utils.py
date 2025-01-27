"""Common utility functions for EMA implementations."""

from __future__ import annotations

from pathlib import Path

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


def p_dot_p(
    t_a: torch.Tensor, gamma_a: torch.Tensor, t_b: torch.Tensor, gamma_b: torch.Tensor
) -> torch.Tensor:
    """Compute p_dot_p value for EMA synthesis.

    Args:
        t_a: First timestep
        gamma_a: First gamma value
        t_b: Second timestep
        gamma_b: Second gamma value

    Returns:
        Tensor: p_dot_p value
    """
    # Handle t=0 case: if both times are 0, ratio is 1
    t_ratio = torch.where(
        (t_a == 0) & (t_b == 0),
        torch.ones_like(t_a),
        t_a / torch.where(t_b == 0, torch.ones_like(t_b), t_b),
    )

    t_exp = torch.where(t_a < t_b, gamma_b, -gamma_a)
    t_max = torch.maximum(t_a, t_b)

    # Handle t=0 case: if both times are 0, max is 1
    t_max = torch.where((t_a == 0) & (t_b == 0), torch.ones_like(t_max), t_max)

    num = (gamma_a + 1) * (gamma_b + 1) * t_ratio**t_exp
    den = (gamma_a + gamma_b + 1) * t_max

    return num / den


def solve_weights(
    gammas: torch.Tensor,
    timesteps: torch.Tensor,
    target_gamma: float,
    *,
    calculation_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Solve for weights that produce target gamma when applied to gammas.

    Args:
        gammas: Tensor of gamma values
        timesteps: Tensor of timesteps
        target_gamma: Target gamma value
        calculation_dtype: Data type for calculations (default=torch.float32)

    Returns:
        Tensor of weights
    """
    # Convert inputs to calculation dtype
    gammas = gammas.to(calculation_dtype)
    timesteps = timesteps.to(calculation_dtype)
    target_gamma = torch.tensor(
        target_gamma, dtype=calculation_dtype, device=gammas.device
    )
    target_timestep = timesteps[-1]  # Use last timestep as target

    # Pre-allocate tensor in calculation dtype
    p_dot_p_matrix = torch.empty(
        (len(gammas), len(gammas)), dtype=calculation_dtype, device=gammas.device
    )

    # Compute p_dot_p matrix
    for i in range(len(gammas)):
        for j in range(len(gammas)):
            p_dot_p_matrix[i, j] = p_dot_p(
                timesteps[i], gammas[i], timesteps[j], gammas[j]
            )

    # Compute target vector
    target_vector = torch.tensor(
        [
            p_dot_p(timesteps[i], gammas[i], target_timestep, target_gamma)
            for i in range(len(gammas))
        ],
        dtype=calculation_dtype,
        device=gammas.device,
    )

    # Solve for weights
    return torch.linalg.solve(p_dot_p_matrix, target_vector)


def _safe_torch_load(path: str | Path, *, map_location=None):
    """Helper function to load checkpoints with weights_only if supported."""
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        # Older PyTorch versions don't support weights_only
        return torch.load(path, map_location=map_location)
