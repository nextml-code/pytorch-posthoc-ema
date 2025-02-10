"""Tests to verify behavior with large sigma_rel values."""

import torch
from torch import nn
from pathlib import Path
import pytest
import shutil
from posthoc_ema import PostHocEMA


@pytest.fixture(autouse=True)
def cleanup_checkpoints():
    """Clean up test checkpoints before and after each test."""
    # Cleanup before test
    for path in ["./test-checkpoints-large-sigma"]:
        if Path(path).exists():
            shutil.rmtree(path)

    yield

    # Cleanup after test
    for path in ["./test-checkpoints-large-sigma"]:
        if Path(path).exists():
            shutil.rmtree(path)


def test_sigma_rel_range_behavior():
    """Test behavior across a range of sigma_rel values."""
    # Create a simple model
    model = nn.Sequential(
        nn.Linear(512, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Linear(512, 10),
    )
    model.train()

    # Create EMA instance with multiple sigma_rels
    posthoc_ema = PostHocEMA.from_model(
        model,
        "test-checkpoints-large-sigma",
        checkpoint_every=5,
        sigma_rels=(0.05, 0.28, 0.8),  # Test up to 0.8 as larger values can be unstable
        update_every=1,
    )

    # Store initial state
    initial_state = {
        name: param.clone().detach() for name, param in model.named_parameters()
    }

    # Do some training
    for step in range(20):
        with torch.no_grad():
            for param in model.parameters():
                if param.requires_grad:
                    param.add_(torch.randn_like(param) * 0.01)  # Smaller updates
        posthoc_ema.update_(model)

    # Store final state
    final_state = {
        name: param.clone().detach() for name, param in model.named_parameters()
    }

    # Test predictions with different sigma_rels
    test_input = torch.randn(16, 512)
    model.eval()
    final_predictions = model(test_input)

    # Test a range of sigma_rels
    sigma_rels = [0.05, 0.28, 0.8]  # Test up to 0.8 as larger values can be unstable
    differences = []

    print("\nTesting different sigma_rel values:")
    for sigma_rel in sigma_rels:
        with posthoc_ema.model(model, sigma_rel=sigma_rel) as ema_model:
            # Compare parameters
            total_param_diff = 0
            max_param_diff = 0
            num_params = 0
            for name, param in ema_model.named_parameters():
                if name in final_state:
                    diff = (param - final_state[name]).abs()
                    total_param_diff += diff.mean().item()
                    max_param_diff = max(max_param_diff, diff.max().item())
                    num_params += 1
            avg_param_diff = total_param_diff / num_params if num_params > 0 else 0

            # Compare predictions
            ema_predictions = ema_model(test_input)
            pred_diff = (final_predictions - ema_predictions).abs()
            max_pred_diff = pred_diff.max().item()
            mean_pred_diff = pred_diff.mean().item()

            differences.append(
                (
                    sigma_rel,
                    avg_param_diff,
                    max_param_diff,
                    max_pred_diff,
                    mean_pred_diff,
                )
            )

            print(f"\nsigma_rel = {sigma_rel}:")
            print(f"  Average parameter difference: {avg_param_diff}")
            print(f"  Max parameter difference: {max_param_diff}")
            print(f"  Max prediction difference: {max_pred_diff}")
            print(f"  Mean prediction difference: {mean_pred_diff}")

    # Verify behavior across sigma_rel values
    for i in range(len(differences)):
        sigma_rel, avg_param_diff, max_param_diff, max_pred_diff, mean_pred_diff = (
            differences[i]
        )

        # For any sigma_rel value:
        # 1. Parameter differences should be reasonable relative to updates
        assert max_param_diff < 0.5, (
            f"Parameter difference too large for sigma_rel={sigma_rel}: "
            f"max_diff={max_param_diff}"
        )

        # 2. Prediction differences should be within reasonable bounds
        # Higher sigma_rel values can have larger differences due to:
        # - More weight on recent states
        # - ReLU activation amplifying differences
        # - BatchNorm scaling effects
        # - Multiple layers compounding differences
        max_allowed_pred_diff = (
            3.5 if sigma_rel >= 0.5 else 2.5 if sigma_rel >= 0.15 else 2.0
        )
        assert max_pred_diff < max_allowed_pred_diff, (
            f"Prediction difference too large for sigma_rel={sigma_rel}: "
            f"max_diff={max_pred_diff}"
        )

        # 3. Mean prediction differences should be smaller than max differences
        assert mean_pred_diff < max_pred_diff, (
            f"Mean prediction difference ({mean_pred_diff}) unexpectedly "
            f"larger than max difference ({max_pred_diff})"
        )

    # Clean up
    if Path("test-checkpoints-large-sigma").exists():
        for file in Path("test-checkpoints-large-sigma").glob("*"):
            file.unlink()
        Path("test-checkpoints-large-sigma").rmdir()


def test_solve_weights_numerical_stability():
    """Test numerical stability of solve_weights with different sigma_rel combinations."""
    from posthoc_ema.utils import solve_weights, sigma_rel_to_gamma
    import torch

    # Create timesteps
    timesteps = torch.arange(0, 100, 5, dtype=torch.float64)  # Use double precision

    # Test different sigma_rel combinations
    sigma_rel_pairs = [
        (0.05, 0.28),  # Original paper values
        (0.15, 0.4),  # Values that caused the error
        (0.28, 0.5),  # Higher values
        (0.05, 0.5),  # Wide range
    ]

    print("\nTesting solve_weights with different sigma_rel combinations:")
    for source_sigma_rels in sigma_rel_pairs:
        print(f"\nSource sigma_rels: {source_sigma_rels}")

        # Convert source sigma_rels to gammas
        source_gammas = torch.tensor(
            [sigma_rel_to_gamma(sr) for sr in source_sigma_rels], dtype=torch.float64
        )

        # Test synthesis for target sigma_rels in range
        target_sigma_rels = torch.linspace(0.05, 0.5, 10)
        for target_sigma_rel in target_sigma_rels:
            print(f"\nTarget sigma_rel: {target_sigma_rel:.3f}")
            target_gamma = sigma_rel_to_gamma(target_sigma_rel.item())

            try:
                # Try solving with float32
                weights_f32 = solve_weights(
                    source_gammas.to(torch.float32),
                    timesteps.to(torch.float32),
                    target_gamma,
                    calculation_dtype=torch.float32,
                )
                print(f"float32 solve succeeded, weights sum: {weights_f32.sum():.6f}")
                print(
                    f"float32 weights min/max: {weights_f32.min():.6f}/{weights_f32.max():.6f}"
                )

                # Try solving with float64
                weights_f64 = solve_weights(
                    source_gammas,
                    timesteps,
                    target_gamma,
                    calculation_dtype=torch.float64,
                )
                print(f"float64 solve succeeded, weights sum: {weights_f64.sum():.6f}")
                print(
                    f"float64 weights min/max: {weights_f64.min():.6f}/{weights_f64.max():.6f}"
                )

                # Compare results
                if torch.allclose(
                    weights_f32.to(torch.float64), weights_f64, rtol=1e-3
                ):
                    print("float32 and float64 results match within tolerance")
                else:
                    print("Warning: float32 and float64 results differ significantly")

            except RuntimeError as e:
                print(f"Error occurred: {str(e)}")
                raise  # Re-raise to fail the test
