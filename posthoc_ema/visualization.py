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
    beta_to_sigma_rel,
    p_dot_p,
    sigma_rel_to_beta,
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
    betas: tuple[float, ...] | None = None,
    sigma_rels: tuple[float, ...] | None = None,
    target_beta_range: tuple[float, float] | None = None,
    target_sigma_rel_range: tuple[float, float] | None = None,
    num_target_points: int = 100,
    max_checkpoints: int = 20,
    update_every: int = 10,
    checkpoint_every: int = 1000,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Compute reconstruction errors for different target values.

    This computes how well different target EMA profiles can be reconstructed
    using a given set of source betas/sigma_rels.

    Args:
        betas: Source EMA decay rates (alternative to sigma_rels)
        sigma_rels: Source relative standard deviations
        target_beta_range: Range of target beta values to evaluate (from_beta, to_beta)
        target_sigma_rel_range: Range of target sigma_rel values to evaluate (min, max)
        num_target_points: Number of target points to evaluate
        max_checkpoints: Maximum number of checkpoints to use
        update_every: Number of steps between EMA updates
        checkpoint_every: Number of steps between checkpoints

    Returns:
        Tuple of (target_sigma_rels, errors, target_betas) where:
            - target_sigma_rels: Tensor of target sigma_rel values
            - errors: Tensor of reconstruction errors for each target sigma_rel
            - target_betas: Tensor of target beta values if target_beta_range was used

    Raises:
        ValueError: If parameters are invalid
    """
    # Validate inputs
    if betas is None and sigma_rels is None:
        sigma_rels = (0.05, 0.28)  # Default values from paper
    if betas is not None and sigma_rels is not None:
        raise ValueError("Cannot specify both betas and sigma_rels")
    
    # Validate beta values
    if betas is not None:
        for beta in betas:
            if not 0 < beta < 1:
                raise ValueError(f"All beta values must be between 0 and 1, got {beta}")
    
    # Validate sigma_rel values
    if sigma_rels is not None:
        for sigma_rel in sigma_rels:
            if sigma_rel <= 0:
                raise ValueError(f"All sigma_rel values must be positive, got {sigma_rel}")
    
    # Convert betas to sigma_rels if needed
    if betas is not None:
        sigma_rels = tuple(beta_to_sigma_rel(beta) for beta in betas)

    # Convert source sigma_rels to gammas
    source_gammas = torch.tensor([sigma_rel_to_gamma(sr) for sr in sigma_rels], dtype=torch.float64)

    # Handle target range
    target_betas = None
    if target_beta_range is not None and target_sigma_rel_range is not None:
        raise ValueError("Cannot specify both target_beta_range and target_sigma_rel_range")
    elif target_beta_range is not None:
        # Validate target beta range
        from_beta, to_beta = target_beta_range
        if not 0 < from_beta < 1:
            raise ValueError(f"from_beta must be between 0 and 1, got {from_beta}")
        if not 0 < to_beta < 1:
            raise ValueError(f"to_beta must be between 0 and 1, got {to_beta}")
        if from_beta >= to_beta:
            raise ValueError(f"from_beta must be less than to_beta, got {from_beta} >= {to_beta}")
        
        print(f"\nTarget beta range: {from_beta} to {to_beta}")
        
        # Generate target beta values
        target_betas = torch.linspace(from_beta, to_beta, num_target_points, dtype=torch.float64)
        print(f"Generated target betas: {target_betas[:5]} ... {target_betas[-5:]}")
        
        # Convert betas to sigma_rels
        target_sigma_rels = torch.tensor([beta_to_sigma_rel(beta.item()) for beta in target_betas])
        print(f"Converted to sigma_rels: {target_sigma_rels[:5]} ... {target_sigma_rels[-5:]}")
    else:
        # Use target_sigma_rel_range (default if neither is specified)
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
        
        # Generate target sigma_rel values
        target_sigma_rels = torch.linspace(min_sigma_rel, max_sigma_rel, num_target_points, dtype=torch.float64)

    # Convert target sigma_rels to gammas for error computation
    target_gammas = torch.tensor([sigma_rel_to_gamma(sr.item()) for sr in target_sigma_rels])
    print(f"Converted to gammas: {target_gammas[:5]} ... {target_gammas[-5:]}")

    # Create timesteps for source profile
    total_steps = max_checkpoints * checkpoint_every
    t_i = torch.arange(0, total_steps + 1, checkpoint_every, dtype=torch.float64)
    
    # Create dense time grid for error computation (reduced from 1000 to 100 points)
    t_dense = torch.linspace(0, total_steps, 100, dtype=torch.float64)

    # Print debug info
    print(f"\nSource sigma_rels: {sigma_rels}")
    print(f"Source gammas: {[g.item() for g in source_gammas]}")
    print(f"Number of checkpoints: {len(t_i)}")
    print(f"Time range: [0, {total_steps}]")

    # Compute error for each target gamma value
    errors = []
    for i, target_gamma in enumerate(target_gammas):
        # Create target profile with constant gamma
        target_gamma_i = target_gamma.expand_as(t_i)
        target_profile = compute_ema_profile(t_i, target_gamma_i, t_dense, target_gamma)
        
        # Skip if target profile is invalid
        if torch.any(torch.isnan(target_profile)) or torch.any(torch.isinf(target_profile)):
            print(f"\nSkipping target {i} (gamma={target_gamma.item():.2f}) - invalid target profile")
            errors.append(1e6)
            continue
            
        # Normalize target profile
        max_val = torch.max(torch.abs(target_profile))
        if max_val > 0:
            target_profile = target_profile / max_val
        
        # Create synthesized profile with subset of checkpoints
        indices = np.linspace(0, len(t_i) - 1, max_checkpoints, dtype=int)
        t_subset = t_i[indices]
        
        # Try each source gamma and take the best result
        min_mse = 1e6  # Start with large but finite error
        best_source_gamma = None
        for source_gamma in source_gammas:
            try:
                # Solve for optimal weights
                gamma_subset = source_gamma.expand_as(t_subset)
                weights = solve_weights(t_subset, gamma_subset, t_dense.reshape(-1, 1), target_gamma.reshape(1))

                # Skip if weights are invalid
                if torch.any(torch.isnan(weights)) or torch.any(torch.isinf(weights)):
                    continue
                
                # Normalize weights
                weights = weights / weights.sum()
                
                # Compute synthesized profile
                synth_profile = torch.zeros_like(t_dense, dtype=torch.float64)
                valid_profile = True
                
                for t, g, w in zip(t_subset, gamma_subset, weights.squeeze(-1)):
                    p = p_dot_p(
                        t_dense.reshape(-1, 1),
                        target_gamma.expand_as(t_dense).reshape(-1, 1),
                        t.expand_as(t_dense).reshape(-1, 1),
                        g.expand_as(t_dense).reshape(-1, 1)
                    ).squeeze()
                    
                    # Skip if p contains invalid values
                    if torch.any(torch.isnan(p)) or torch.any(torch.isinf(p)):
                        valid_profile = False
                        break
                    
                    # Add weighted contribution
                    synth_profile += w * p
                
                if not valid_profile:
                    continue
                
                # Normalize synthesized profile
                max_val = torch.max(torch.abs(synth_profile))
                if max_val > 0:
                    synth_profile = synth_profile / max_val
                    
                    mse = torch.mean((target_profile - synth_profile) ** 2).item()
                    if mse < min_mse:
                        min_mse = mse
                        best_source_gamma = source_gamma.item()

            except RuntimeError:
                continue
        
        if best_source_gamma is not None:
            print(f"Target {i} (gamma={target_gamma.item():.2f}) - Best error: {min_mse:.2e} with source gamma={best_source_gamma:.2f}")
        else:
            print(f"\nSkipping target {i} (gamma={target_gamma.item():.2f}) - no valid solution found")
        errors.append(min_mse)

    return target_sigma_rels, torch.tensor(errors, dtype=torch.float64), target_betas


def plot_reconstruction_errors(
    target_sigma_rels: torch.Tensor,
    errors: torch.Tensor,
    source_sigma_rels: tuple[float, ...] | None = None,
    source_betas: tuple[float, ...] | None = None,
    target_betas: torch.Tensor | None = None,
    title: str | None = None,
) -> Image.Image:
    """Plot reconstruction errors vs target beta values.

    Args:
        target_sigma_rels: Target sigma_rel values
        errors: Reconstruction errors for each target sigma_rel
        source_sigma_rels: Source sigma_rel values (alternative to source_betas)
        source_betas: Source beta values (alternative to source_sigma_rels)
        target_betas: Target beta values (if None, will be computed from target_sigma_rels)
        title: Optional title for the plot

    Returns:
        PIL Image containing the plot

    Raises:
        ImportError: If matplotlib is not installed
    """
    _check_matplotlib()

    if source_betas is None and source_sigma_rels is None:
        raise ValueError("Must specify either source_betas or source_sigma_rels")
    if source_betas is not None and source_sigma_rels is not None:
        raise ValueError("Cannot specify both source_betas and source_sigma_rels")

    # Convert source sigma_rels to betas if needed
    if source_sigma_rels is not None:
        source_betas = [sigma_rel_to_beta(sr) for sr in source_sigma_rels]

    # Convert target sigma_rels to betas if not provided
    if target_betas is None:
        target_betas = torch.tensor([sigma_rel_to_beta(sr.item()) for sr in target_sigma_rels])

    print(f"\nPlotting with target betas: {target_betas[:5]} ... {target_betas[-5:]}")
    print(f"Source betas: {source_betas}")
    print(f"Target betas shape: {target_betas.shape}")
    print(f"Errors shape: {errors.shape}")
    print(f"Target betas range: [{target_betas.min().item():.4f}, {target_betas.max().item():.4f}]")

    # Create figure and axis with constrained layout
    plt.rcParams['figure.constrained_layout.use'] = True
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot reconstruction error vs target beta
    ax.semilogy(target_betas.numpy(), errors.numpy(), marker="o")
    ax.grid(True)

    # Add vertical lines for source betas
    for beta in source_betas:
        ax.axvline(beta, color="red", linestyle="--", alpha=0.5)

    # Set labels
    ax.set_xlabel("Target β")
    ax.set_ylabel("Reconstruction Error")

    # Add title showing source parameters
    if title is None:
        title = f"Reconstruction Error vs Target β (Source β={tuple(f'{beta:.4f}' for beta in source_betas)})"
    ax.set_title(title)
    
    # Set x-axis limits to match the actual target beta range with a small margin
    beta_range = target_betas[-1].item() - target_betas[0].item()
    margin = beta_range * 0.02  # 2% margin
    xlim_min = target_betas[0].item() - margin
    xlim_max = target_betas[-1].item() + margin
    ax.set_xlim(xlim_min, xlim_max)
    print(f"Setting x-axis limits to: [{xlim_min:.4f}, {xlim_max:.4f}]")

    # Convert figure to PIL Image
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)
    
    return img


def reconstruction_error(
    betas: tuple[float, ...] | None = None,
    sigma_rels: tuple[float, ...] | None = None,
    target_beta_range: tuple[float, float] | None = None,
    target_sigma_rel_range: tuple[float, float] | None = None,
    num_target_points: int = 100,
    max_checkpoints: int = 20,
    update_every: int = 10,
    checkpoint_every: int = 1000,
    title: Optional[str] = None,
) -> Image.Image:
    """Plot reconstruction error for different target values.

    This helps visualize how well different target EMA profiles can be reconstructed
    using a given set of source betas/sigma_rels. Use this before training to choose
    appropriate EMA parameters.

    Args:
        betas: Source EMA decay rates (alternative to sigma_rels)
        sigma_rels: Source relative standard deviations
        target_beta_range: Range of target beta values to evaluate (from_beta, to_beta)
        target_sigma_rel_range: Range of target sigma_rel values to evaluate (min, max)
        num_target_points: Number of target points to evaluate
        max_checkpoints: Maximum number of checkpoints to use
        update_every: Number of steps between EMA updates
        checkpoint_every: Number of steps between checkpoints
        title: Optional title for the plot

    Returns:
        PIL Image containing the plot
    """
    target_sigma_rels, errors, target_betas = compute_reconstruction_errors(
        betas=betas,
        sigma_rels=sigma_rels,
        target_beta_range=target_beta_range,
        target_sigma_rel_range=target_sigma_rel_range,
        num_target_points=num_target_points,
        max_checkpoints=max_checkpoints,
        update_every=update_every,
        checkpoint_every=checkpoint_every,
    )

    return plot_reconstruction_errors(
        target_sigma_rels=target_sigma_rels,
        errors=errors,
        source_betas=betas,
        source_sigma_rels=sigma_rels,
        target_betas=target_betas,
        title=title,
    ) 