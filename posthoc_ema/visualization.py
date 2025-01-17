"""Visualization utilities for post-hoc EMA.

This module provides functions for visualizing the reconstruction error between
target EMA profiles and synthesized profiles, as well as how the error changes
with different numbers of checkpoints.
"""

from io import BytesIO
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image

from posthoc_ema.utils import (
    p_dot_p,
    sigma_rel_to_gamma,
    solve_weights,
)

try:
    from matplotlib import pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

def _check_matplotlib():
    if not HAS_MATPLOTLIB:
        raise ImportError(
            "matplotlib is required for visualization functions. "
            "Please install it with: pip install matplotlib"
        )

def compute_ema_profile(
    t_i: torch.Tensor,
    gamma_i: torch.Tensor,
    t_eval: torch.Tensor,
    gamma_eval: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute EMA profile at evaluation points.
    
    Args:
        t_i: Timesteps for the profile
        gamma_i: Gamma values for the profile
        t_eval: Points at which to evaluate the profile
        gamma_eval: Optional gamma value for evaluation points (defaults to gamma_i[0])
        
    Returns:
        Profile values at evaluation points
    """
    # Reshape tensors for broadcasting
    t_i = t_i.reshape(-1, 1)  # [N, 1]
    gamma_i = gamma_i.reshape(-1, 1)  # [N, 1]
    t_eval = t_eval.reshape(-1, 1)  # [M, 1]
    
    # Use provided gamma_eval or default to gamma_i[0]
    if gamma_eval is None:
        gamma_eval = gamma_i[0]
    gamma_eval = gamma_eval.reshape(-1, 1)  # [1, 1]
    
    # Compute profile values
    profile = torch.zeros(len(t_eval), dtype=torch.float64)
    for t, g in zip(t_i, gamma_i):
        p = p_dot_p(
            t_eval,  # [M, 1]
            gamma_eval.expand_as(t_eval),  # [M, 1]  # Use target gamma for first argument
            t.expand_as(t_eval),  # [M, 1]
            g.expand_as(t_eval),  # [M, 1]  # Use source gamma for second argument
        ).squeeze()
        profile += p
    
    return profile


def compute_reconstruction_errors(
    sigma_rels: tuple[float, ...],
    target_sigma_rel_range: tuple[float, float] | None = None,
    num_target_points: int = 100,
    max_checkpoints: int = 20,
    update_every: int = 10,
    checkpoint_every: int = 1000,
) -> tuple[torch.Tensor, torch.Tensor, None]:
    """Compute reconstruction errors for different target values.

    This computes how well different target EMA profiles can be reconstructed
    using a given set of source sigma_rels, following the approach in the paper.
    The error should be minimal around the source sigma_rels, as these profiles
    can be reconstructed exactly.

    Args:
        sigma_rels: Source relative standard deviations
        target_sigma_rel_range: Range of target sigma_rel values to evaluate (min, max)
        num_target_points: Number of target points to evaluate
        max_checkpoints: Maximum number of checkpoints to use
        update_every: Number of steps between EMA updates
        checkpoint_every: Number of steps between checkpoints

    Returns:
        Tuple of (target_sigma_rels, errors, None) where:
            - target_sigma_rels: Tensor of target sigma_rel values
            - errors: Tensor of reconstruction errors for each target sigma_rel
            - None: For backward compatibility
    """
    # Validate sigma_rel values
    for sigma_rel in sigma_rels:
        if sigma_rel <= 0:
            raise ValueError(f"All sigma_rel values must be positive, got {sigma_rel}")

    # Convert source sigma_rels to gammas
    source_gammas = torch.tensor([sigma_rel_to_gamma(sr) for sr in sigma_rels], dtype=torch.float64)

    # Use target_sigma_rel_range (default if not specified)
    if target_sigma_rel_range is None:
        target_sigma_rel_range = (0.05, 0.28)  # Default values from paper
    
    # Validate target sigma_rel range
    min_sigma_rel, max_sigma_rel = target_sigma_rel_range
    if min_sigma_rel <= 0:
        raise ValueError(f"min_sigma_rel must be positive, got {min_sigma_rel}")
    if max_sigma_rel <= 0:
        raise ValueError(f"max_sigma_rel must be positive, got {max_sigma_rel}")
    if min_sigma_rel >= max_sigma_rel:
        raise ValueError(f"min_sigma_rel must be less than max_sigma_rel, got {min_sigma_rel} >= {max_sigma_rel}")
    
    # Generate target sigma_rel values, ensuring source values are included
    target_points = []
    for sr in sorted(sigma_rels):
        if sr >= min_sigma_rel and sr <= max_sigma_rel:
            target_points.append(sr)
    
    # Add evenly spaced points, excluding points too close to source values
    num_extra_points = num_target_points - len(target_points)
    extra_points = torch.linspace(min_sigma_rel, max_sigma_rel, num_extra_points + 20)
    for p in extra_points:
        p = p.item()
        # Skip if too close to a source value
        if any(abs(p - sr) < (max_sigma_rel - min_sigma_rel) / num_target_points for sr in sigma_rels):
            continue
        target_points.append(p)
        if len(target_points) >= num_target_points:
            break
    
    target_sigma_rels = torch.tensor(sorted(target_points[:num_target_points]), dtype=torch.float64)

    # Convert target sigma_rels to gammas for error computation
    target_gammas = torch.tensor([sigma_rel_to_gamma(sr.item()) for sr in target_sigma_rels])

    # Create timesteps for source profile
    total_steps = max_checkpoints * checkpoint_every
    t_i = torch.arange(0, total_steps + 1, checkpoint_every, dtype=torch.float64)
    
    # Create dense time grid for error computation
    t_dense = torch.linspace(0, total_steps, 100, dtype=torch.float64)

    # Compute error for each target gamma value
    errors = []
    for i, target_gamma in enumerate(target_gammas):
        # Create target profile with constant gamma
        target_gamma_i = target_gamma.expand_as(t_i)
        target_profile = compute_ema_profile(t_i, target_gamma_i, t_dense, target_gamma)
        
        # Skip if target profile is invalid
        if torch.any(torch.isnan(target_profile)) or torch.any(torch.isinf(target_profile)):
            errors.append(1e6)
            continue
            
        # Normalize target profile
        max_val = torch.max(torch.abs(target_profile))
        if max_val > 0:
            target_profile = target_profile / max_val

        # Create synthesized profile with subset of checkpoints
        indices = np.linspace(0, len(t_i) - 1, max_checkpoints, dtype=int)
        t_subset = t_i[indices]
        
        # Create source profiles and solve for optimal weights
        try:
            # Create matrix of source profiles
            source_profiles = []
            for source_gamma in source_gammas:
                gamma_subset = source_gamma.expand_as(t_subset)
                profile = compute_ema_profile(t_subset, gamma_subset, t_dense, target_gamma)
                if torch.max(torch.abs(profile)) > 0:
                    profile = profile / torch.max(torch.abs(profile))
                source_profiles.append(profile)
            
            # Stack profiles and solve for weights
            A = torch.stack(source_profiles, dim=1)  # [time, sources]
            b = target_profile.unsqueeze(1)  # [time, 1]
            try:
                # Solve least squares problem
                weights = torch.linalg.lstsq(A, b).solution.squeeze()
                
                # Compute synthesized profile
                synth_profile = A @ weights
                
                # Compute MSE
                mse = torch.mean((target_profile - synth_profile) ** 2)
                
            except Exception:
                mse = 1e6
                
        except Exception as e:
            mse = 1e6
        
        # Store error
        errors.append(mse)
    
    # Convert errors to tensor
    errors = torch.tensor(errors, dtype=torch.float64)
    
    return target_sigma_rels, errors, None


def plot_reconstruction_errors(
    target_sigma_rels: torch.Tensor,
    errors: torch.Tensor,
    source_sigma_rels: tuple[float, ...],
    title: str | None = None,
    figsize: tuple[int, int] = (10, 6),
) -> Image.Image:
    """Plot reconstruction errors for different target sigma_rel values.
    
    Args:
        target_sigma_rels: Target sigma_rel values
        errors: Reconstruction errors for each target sigma_rel
        source_sigma_rels: Source sigma_rel values used for reconstruction
        title: Optional title for the plot
        figsize: Figure size in inches
        
    Returns:
        PIL Image of the plot
    """
    _check_matplotlib()
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Plot errors
    plt.semilogy(target_sigma_rels, errors)
    
    # Add vertical lines for source sigma_rels
    for sr in source_sigma_rels:
        plt.axvline(sr, color='r', linestyle='--', alpha=0.5)
    
    # Add labels and title
    plt.xlabel('Target sigma_rel')
    plt.ylabel('Reconstruction Error')
    if title:
        plt.title(title)
    else:
        plt.title('EMA Profile Reconstruction Error')
    
    # Add grid
    plt.grid(True)
    
    # Save to buffer
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    
    # Convert to PIL Image
    buf.seek(0)
    return Image.open(buf)


def reconstruction_error(
    sigma_rels: tuple[float, ...],
    target_sigma_rel_range: tuple[float, float] | None = None,
    num_target_points: int = 100,
    max_checkpoints: int = 20,
    update_every: int = 10,
    checkpoint_every: int = 1000,
    title: str | None = None,
    figsize: tuple[int, int] = (10, 6),
) -> Image.Image:
    """Compute and plot reconstruction errors for different target sigma_rel values.
    
    This is a convenience function that combines compute_reconstruction_errors and
    plot_reconstruction_errors.
    
    Args:
        sigma_rels: Source relative standard deviations
        target_sigma_rel_range: Range of target sigma_rel values to evaluate (min, max)
        num_target_points: Number of target points to evaluate
        max_checkpoints: Maximum number of checkpoints to use
        update_every: Number of steps between EMA updates
        checkpoint_every: Number of steps between checkpoints
        title: Optional title for the plot
        figsize: Figure size in inches
        
    Returns:
        PIL Image of the reconstruction error plot
    """
    # Compute errors
    target_sigma_rels, errors, _ = compute_reconstruction_errors(
        sigma_rels=sigma_rels,
        target_sigma_rel_range=target_sigma_rel_range,
        num_target_points=num_target_points,
        max_checkpoints=max_checkpoints,
        update_every=update_every,
        checkpoint_every=checkpoint_every,
    )
    
    # Plot errors
    return plot_reconstruction_errors(
        target_sigma_rels=target_sigma_rels,
        errors=errors,
        source_sigma_rels=sigma_rels,
        title=title,
        figsize=figsize,
    ) 