from copy import deepcopy
from pathlib import Path

import torch

from posthoc_ema import PostHocEMA


def test_basic_usage_with_updates():
    """Test the basic usage pattern with model updates."""
    model = torch.nn.Linear(512, 512)
    posthoc_ema = PostHocEMA.from_model(
        model, 
        "posthoc-ema",
        checkpoint_every=5,  # More frequent checkpoints for testing
        sigma_rels=(0.05, 0.28),  # Explicitly set sigma_rels
    )

    # Simulate training loop
    for _ in range(10):  # Reduced from 1000 for test speed
        # mutate your network, normally with an optimizer
        with torch.no_grad():
            model.weight.copy_(torch.randn_like(model.weight))
            model.bias.copy_(torch.randn_like(model.bias))
        posthoc_ema.update_(model)

    # Verify we can get predictions
    data = torch.randn(1, 512)
    predictions = model(data)
    assert predictions.shape == (1, 512)


def test_context_manager_helper():
    """Test using the context manager helper for EMA model."""
    model = torch.nn.Linear(512, 512)
    posthoc_ema = PostHocEMA.from_model(
        model, 
        "posthoc-ema",
        checkpoint_every=5,  # More frequent checkpoints for testing
        sigma_rels=(0.05, 0.28),  # Explicitly set sigma_rels
    )

    # Update enough times to create checkpoints
    for _ in range(10):  # Reduced from 1000 for test speed
        with torch.no_grad():
            model.weight.copy_(torch.randn_like(model.weight))
            model.bias.copy_(torch.randn_like(model.bias))
        posthoc_ema.update_(model)

    data = torch.randn(1, 512)
    predictions = model(data)

    # use the helper
    with posthoc_ema.model(model, sigma_rel=0.15) as ema_model:
        ema_predictions = ema_model(data)
        assert ema_predictions.shape == predictions.shape


def test_manual_cpu_usage():
    """Test manual CPU usage without the context manager."""
    model = torch.nn.Linear(512, 512)
    posthoc_ema = PostHocEMA.from_model(
        model, 
        "posthoc-ema",
        checkpoint_every=5,  # More frequent checkpoints for testing
        sigma_rels=(0.05, 0.28),  # Explicitly set sigma_rels
    )

    # Update enough times to create checkpoints
    for _ in range(10):  # Reduced from 1000 for test speed
        with torch.no_grad():
            model.weight.copy_(torch.randn_like(model.weight))
            model.bias.copy_(torch.randn_like(model.bias))
        posthoc_ema.update_(model)

    data = torch.randn(1, 512)

    # or without magic
    model.cpu()
    with posthoc_ema.state_dict(sigma_rel=0.15) as state_dict:
        ema_model = deepcopy(model)
        ema_model.load_state_dict(state_dict)
        ema_predictions = ema_model(data)
        assert ema_predictions.shape == (1, 512)
        del ema_model


def test_synthesize_after_training():
    """Test synthesizing EMA after training."""
    model = torch.nn.Linear(512, 512)
    
    # First create some checkpoints
    posthoc_ema = PostHocEMA.from_model(
        model, 
        "posthoc-ema",
        checkpoint_every=5,  # More frequent checkpoints for testing
        sigma_rels=(0.05, 0.28),  # Explicitly set sigma_rels
    )
    for _ in range(10):  # Reduced from 1000 for test speed
        with torch.no_grad():
            model.weight.copy_(torch.randn_like(model.weight))
            model.bias.copy_(torch.randn_like(model.bias))
        posthoc_ema.update_(model)

    data = torch.randn(1, 512)

    # Synthesize after training
    posthoc_ema = PostHocEMA.from_path("posthoc-ema", model)
    with posthoc_ema.model(model, sigma_rel=0.15) as ema_model:
        ema_predictions = ema_model(data)
        assert ema_predictions.shape == (1, 512)


def test_synthesize_without_model():
    """Test synthesizing EMA without model."""
    model = torch.nn.Linear(512, 512)
    
    # First create some checkpoints
    posthoc_ema = PostHocEMA.from_model(
        model, 
        "posthoc-ema",
        checkpoint_every=5,  # More frequent checkpoints for testing
        sigma_rels=(0.05, 0.28),  # Explicitly set sigma_rels
    )
    for _ in range(10):  # Reduced from 1000 for test speed
        with torch.no_grad():
            model.weight.copy_(torch.randn_like(model.weight))
            model.bias.copy_(torch.randn_like(model.bias))
        posthoc_ema.update_(model)

    # Or without model
    posthoc_ema = PostHocEMA.from_path("posthoc-ema")
    with posthoc_ema.state_dict(sigma_rel=0.15) as state_dict:
        assert isinstance(state_dict, dict)
        assert len(state_dict) > 0


def test_set_parameters_during_training():
    """Test setting parameters to EMA state during training."""
    model = torch.nn.Linear(512, 512)
    posthoc_ema = PostHocEMA.from_model(
        model, 
        "posthoc-ema",
        checkpoint_every=5,  # More frequent checkpoints for testing
        sigma_rels=(0.05, 0.28),  # Explicitly set sigma_rels
    )

    # Update enough times to create checkpoints
    for _ in range(10):  # Reduced from 1000 for test speed
        with torch.no_grad():
            model.weight.copy_(torch.randn_like(model.weight))
            model.bias.copy_(torch.randn_like(model.bias))
        posthoc_ema.update_(model)

    # Save original state
    original_state = deepcopy(model.state_dict())

    # Set parameters to EMA state during training
    with posthoc_ema.state_dict(sigma_rel=0.15) as state_dict:
        model.load_state_dict(state_dict, strict=False)
        # Verify state has changed
        assert not torch.allclose(model.weight, original_state['weight'].clone().detach())

    # Clean up
    if Path("posthoc-ema").exists():
        for file in Path("posthoc-ema").glob("*"):
            file.unlink()
        Path("posthoc-ema").rmdir() 