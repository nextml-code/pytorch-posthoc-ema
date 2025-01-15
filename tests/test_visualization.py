"""Tests for visualization functions."""

import numpy as np
import pytest
import torch
from matplotlib import pyplot as plt
from PIL import Image

from posthoc_ema.utils import beta_to_sigma_rel, sigma_rel_to_gamma
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
    target_sigma_rels, errors, target_betas = compute_reconstruction_errors(
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
    assert target_betas is None  # No target betas when using sigma_rel range

    # Check values
    assert torch.all(target_sigma_rels >= 0.05)
    assert torch.all(target_sigma_rels <= 0.28)
    assert torch.all(errors >= 0)
    assert torch.any(torch.isfinite(errors))  # At least some errors should be finite


def test_compute_reconstruction_errors_with_sigma_rels():
    """Test compute_reconstruction_errors with sigma_rels."""
    sigma_rels = (0.05, 0.10)  # Values from paper
    target_sigma_rels, errors, target_betas = compute_reconstruction_errors(
        sigma_rels=sigma_rels,
        target_sigma_rel_range=(0.05, 0.28),  # Range from paper
    )
    assert len(target_sigma_rels) == 100  # Default num_target_points
    assert len(errors) == 100
    assert target_betas is None  # No target betas when using sigma_rel range


def test_compute_reconstruction_errors_with_target_beta_range():
    """Test compute_reconstruction_errors with target_beta_range."""
    betas = (0.9, 0.99, 0.999)
    target_sigma_rels, errors, target_betas = compute_reconstruction_errors(
        betas=betas,
        target_beta_range=(0.9, 0.95),
    )
    assert len(target_sigma_rels) == 100  # Default num_target_points
    assert len(errors) == 100
    assert target_betas is not None
    assert len(target_betas) == 100
    assert torch.all(target_betas >= 0.9)
    assert torch.all(target_betas <= 0.95)


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
        sigma_rels=(0.05, 0.10),  # Must specify either betas or sigma_rels
    )
    assert isinstance(img, Image.Image)


def test_reconstruction_error_with_target_beta_range():
    """Test reconstruction_error with target_beta_range."""
    betas = (0.9, 0.99, 0.999)
    img = reconstruction_error(
        betas=betas,
        target_beta_range=(0.9, 0.95),
    )
    assert isinstance(img, Image.Image)


def test_reconstruction_error_invalid_input():
    """Test that reconstruction_error handles invalid input correctly."""
    with pytest.raises(ValueError):
        # Test with both betas and sigma_rels
        reconstruction_error(betas=(0.9, 0.999), sigma_rels=(0.05, 0.28))

    with pytest.raises(ValueError):
        # Test with both target ranges
        reconstruction_error(
            target_beta_range=(0.9, 0.999),
            target_sigma_rel_range=(0.05, 0.28)
        )

    with pytest.raises(ValueError):
        # Test with invalid beta range
        reconstruction_error(target_beta_range=(0.9999, 0.9))  # Wrong order

    with pytest.raises(ValueError):
        # Test with invalid beta value
        reconstruction_error(target_beta_range=(1.1, 0.999))  # Beta > 1

    with pytest.raises(ValueError):
        # Test with invalid sigma_rel range
        reconstruction_error(target_sigma_rel_range=(0.28, 0.05))  # Wrong order

    with pytest.raises(ValueError):
        # Test with negative sigma_rel
        reconstruction_error(sigma_rels=(-0.05, 0.28))

    with pytest.raises(ValueError):
        # Test with invalid beta
        reconstruction_error(betas=(1.1, 0.999))  # Beta > 1


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