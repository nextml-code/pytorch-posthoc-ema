"""Tests for visualization functions."""

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from PIL import Image

from posthoc_ema.posthoc_ema import PostHocEMA
from posthoc_ema.utils import sigma_rel_to_gamma
from posthoc_ema.visualization import (
    compute_reconstruction_errors,
    plot_reconstruction_errors,
    reconstruction_error,
)


def test_compute_reconstruction_errors():
    """Test that compute_reconstruction_errors returns correct shapes and values."""
    # Print source values
    sigma_rels = (0.05, 0.10)  # Values from paper
    print("\nSource sigma_rels:", sigma_rels)
    print("Source gammas:", tuple(map(sigma_rel_to_gamma, sigma_rels)))

    # Compute errors
    target_sigma_rels, errors, _ = compute_reconstruction_errors(
        sigma_rels=sigma_rels,
        target_sigma_rel_range=(0.05, 0.28),  # Range from paper
        num_target_points=50,  # More points for better precision
        max_checkpoints=20,  # More checkpoints for better reconstruction
        update_every=10,
        checkpoint_every=10,
    )

    # Check shapes
    assert len(target_sigma_rels) == 50
    assert len(errors) == 50

    # Check values
    assert torch.all(target_sigma_rels >= 0.05)
    assert torch.all(target_sigma_rels <= 0.28)
    assert torch.all(errors >= 0)
    assert torch.any(torch.isfinite(errors))  # At least some errors should be finite


def test_compute_reconstruction_errors_with_sigma_rels():
    """Test compute_reconstruction_errors with sigma_rels."""
    sigma_rels = (0.05, 0.10)  # Values from paper
    target_sigma_rels, errors, _ = compute_reconstruction_errors(
        sigma_rels=sigma_rels,
        target_sigma_rel_range=(0.05, 0.28),  # Range from paper
    )
    assert len(target_sigma_rels) == 100  # Default num_target_points
    assert len(errors) == 100


def test_plot_reconstruction_errors():
    """Test that plot_reconstruction_errors returns a PIL Image."""
    target_sigma_rels = torch.linspace(0.05, 0.28, 100)
    errors = torch.ones_like(target_sigma_rels)
    img = plot_reconstruction_errors(
        target_sigma_rels=target_sigma_rels,
        errors=errors,
        source_sigma_rels=(0.05, 0.10),
    )
    assert isinstance(img, Image.Image)


def test_reconstruction_error_with_sigma_rels():
    """Test reconstruction_error with sigma_rels."""
    sigma_rels = (0.05, 0.10)  # Values from paper
    img = reconstruction_error(
        sigma_rels=sigma_rels,
        target_sigma_rel_range=(0.05, 0.28),  # Range from paper
    )
    assert isinstance(img, Image.Image)


def test_reconstruction_error_with_defaults():
    """Test reconstruction_error with default values."""
    img = reconstruction_error(
        sigma_rels=(0.05, 0.10),  # Must specify sigma_rels
    )
    assert isinstance(img, Image.Image)


def test_reconstruction_error_invalid_input():
    """Test that reconstruction_error handles invalid input correctly."""
    with pytest.raises(ValueError):
        # Test with invalid sigma_rel range
        reconstruction_error(
            sigma_rels=(0.05, 0.28),
            target_sigma_rel_range=(0.28, 0.05),  # Wrong order
        )

    with pytest.raises(ValueError):
        # Test with negative sigma_rel
        reconstruction_error(sigma_rels=(-0.05, 0.28))


def test_p_dot_p_debug():
    """Test p_dot_p function with simple values."""
    import torch

    from posthoc_ema.utils import p_dot_p
    
    # Create simple test tensors
    t_a = torch.tensor([[0.0, 10.0], [20.0, 30.0]], dtype=torch.float64)
    gamma_a = torch.tensor([[7.0, 7.0], [7.0, 7.0]], dtype=torch.float64)
    t_b = torch.tensor([[0.0, 10.0], [20.0, 30.0]], dtype=torch.float64)
    gamma_b = torch.tensor([[28.0, 28.0], [28.0, 28.0]], dtype=torch.float64)
    
    # Compute dot product
    result = p_dot_p(t_a, gamma_a, t_b, gamma_b)
    
    # Result should be finite
    assert torch.all(torch.isfinite(result)) 


def test_reconstruction_error_at_source_values():
    """Test reconstruction error specifically at source sigma_rel values."""
    source_sigma_rels = (0.05, 0.10)
    target_sigma_rels, errors, _ = compute_reconstruction_errors(
        sigma_rels=source_sigma_rels,
        target_sigma_rel_range=(0.05, 0.28),
        num_target_points=100,
        max_checkpoints=100,  # More checkpoints
        update_every=5,  # More frequent updates
        checkpoint_every=5,  # More frequent checkpoints
    )

    # Find indices closest to source sigma_rels
    errors_at_source = []
    for sr in source_sigma_rels:
        idx = torch.argmin(torch.abs(target_sigma_rels - sr))
        error = errors[idx]
        print(f"\nError at sigma_rel={sr:.3f}: {error:.2e}")
        print(f"Closest evaluated sigma_rel: {target_sigma_rels[idx]:.3f}")
        errors_at_source.append(error)

    # Errors should be small but not necessarily zero due to discretization and numerical precision
    assert all(e < 1e-4 for e in errors_at_source), "Errors at source sigma_rels should be reasonably small" 


def test_posthoc_ema_reconstruction_error(tmp_path: Path):
    """Test that PostHocEMA.reconstruction_error runs without errors."""
    # Create a simple model
    model = nn.Linear(10, 10)
    
    # Initialize PostHocEMA
    posthoc_ema = PostHocEMA.from_model(
        model=model,
        checkpoint_dir=tmp_path,
        max_checkpoints=100,
        update_every=5,
        checkpoint_every=5,
    )
    
    # Create some checkpoints
    for _ in range(50):
        # Random update to model weights
        with torch.no_grad():
            for param in model.parameters():
                param.add_(torch.randn_like(param) * 0.01)
        posthoc_ema.update_(model)
    
    # Call reconstruction_error
    posthoc_ema.reconstruction_error() 