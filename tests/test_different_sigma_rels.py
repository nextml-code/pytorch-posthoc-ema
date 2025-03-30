"""Tests to verify that different sigma_rel values produce different weights."""

import torch
from posthoc_ema import PostHocEMA
from pathlib import Path
import shutil
import pytest


@pytest.fixture(autouse=True)
def cleanup_checkpoints():
    """Clean up test checkpoints before and after each test."""
    # Cleanup before test
    for path in ["./test-checkpoints-diff-sigma"]:
        if Path(path).exists():
            shutil.rmtree(path)

    yield

    # Cleanup after test
    for path in ["./test-checkpoints-diff-sigma"]:
        if Path(path).exists():
            shutil.rmtree(path)


def test_different_sigma_rels_produce_different_weights():
    """Test that different sigma_rel values produce different weights."""
    # Create a simple model
    model = torch.nn.Sequential(
        torch.nn.Linear(512, 512),
        torch.nn.BatchNorm1d(512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 10),
    )
    model.train()

    # Disable gradients for some parameters
    model[0].bias.requires_grad = False
    model[1].weight.requires_grad = False
    model[1].bias.requires_grad = False

    # Print initial model parameters
    print("\nInitial model parameters:")
    for name, param in model.named_parameters():
        print(f"{name}: requires_grad={param.requires_grad}, shape={param.shape}")

    # Create EMA instance
    posthoc_ema = PostHocEMA.from_model(
        model,
        "test-checkpoints-diff-sigma",  # Changed from "posthoc-ema"
        checkpoint_every=5,
        sigma_rels=(0.05, 0.28),  # Use two different sigma_rels
        update_after_step=0,  # Start immediately to match original behavior
    )

    # Do some training to build up EMA weights
    for _ in range(20):  # More updates to ensure difference
        with torch.no_grad():
            # Random updates to weights
            for param in model.parameters():
                param.add_(torch.randn_like(param) * 0.1)
        posthoc_ema.update_(model)

    # Get weights for two different sigma_rels
    with posthoc_ema.state_dict(sigma_rel=0.05) as state_dict_1:
        with posthoc_ema.state_dict(sigma_rel=0.4) as state_dict_2:
            # Compare weights
            max_diff = 0.0
            mean_diff = 0.0
            num_params = 0

            for key in state_dict_1.keys():
                # Skip batch norm running statistics
                if "running_" in key:
                    continue

                if key in state_dict_2:
                    diff = (state_dict_1[key] - state_dict_2[key]).abs()
                    # Skip integer tensors or convert to float for mean calculation
                    if diff.dtype in [torch.int, torch.int8, torch.int16, torch.int32, torch.int64, torch.long]:
                        if "num_batches_tracked" in key:
                            continue  # Skip entirely
                        diff = diff.float()
                    max_diff = max(max_diff, diff.max().item())
                    mean_diff += diff.mean().item()
                    num_params += 1

            mean_diff /= num_params

            print(f"Max difference between weights: {max_diff}")
            print(f"Mean difference between weights: {mean_diff}")

            # Assert weights are different
            assert (
                max_diff > 1e-4
            ), "Weights should be different for different sigma_rels"

            # Also verify that no weights are identical (except running stats)
            for key in state_dict_1.keys():
                # Skip batch norm running statistics and num_batches_tracked
                if "running_" in key or "num_batches_tracked" in key:
                    continue

                if key in state_dict_2:
                    assert not torch.allclose(
                        state_dict_1[key], state_dict_2[key], rtol=1e-5, atol=1e-5
                    ), f"Weights for {key} should be different"

    # Clean up
    if Path("test-checkpoints-diff-sigma").exists():
        for file in Path("test-checkpoints-diff-sigma").glob("*"):
            file.unlink()
        Path("test-checkpoints-diff-sigma").rmdir()


def test_different_sigma_rels_produce_different_predictions():
    """Test that different sigma_rel values produce different predictions."""
    # Create a simple model
    model = torch.nn.Sequential(
        torch.nn.Linear(512, 512),
        torch.nn.BatchNorm1d(512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 10),
    )
    model.train()

    # Disable gradients for some parameters
    model[0].bias.requires_grad = False
    model[1].weight.requires_grad = False
    model[1].bias.requires_grad = False

    # Print initial model parameters
    print("\nInitial model parameters:")
    for name, param in model.named_parameters():
        print(f"{name}: requires_grad={param.requires_grad}, shape={param.shape}")

    # Create EMA instance
    posthoc_ema = PostHocEMA.from_model(
        model,
        "test-checkpoints-diff-sigma",
        checkpoint_every=5,
        sigma_rels=(0.05, 0.28),
        update_after_step=0,  # Start immediately to match original behavior
    )

    # Do some training to build up EMA weights
    for _ in range(20):
        with torch.no_grad():
            for param in model.parameters():
                param.add_(torch.randn_like(param) * 0.1)
        posthoc_ema.update_(model)

    # Create test input
    test_input = torch.randn(16, 512)

    # Get predictions with different sigma_rels
    model.eval()
    with posthoc_ema.model(model, sigma_rel=0.05) as ema_model:
        predictions_1 = ema_model(test_input)

    with posthoc_ema.model(model, sigma_rel=0.4) as ema_model:
        predictions_2 = ema_model(test_input)

    # Compare predictions
    max_diff = (predictions_1 - predictions_2).abs().max().item()
    mean_diff = (predictions_1 - predictions_2).abs().mean().item()

    print(f"Max difference between predictions: {max_diff}")
    print(f"Mean difference between predictions: {mean_diff}")

    # Assert predictions are different
    assert not torch.allclose(
        predictions_1, predictions_2, rtol=1e-5, atol=1e-5
    ), "Predictions should be different for different sigma_rels"

    assert max_diff > 1e-4, "Predictions should be significantly different"

    # Clean up
    if Path("test-checkpoints-diff-sigma").exists():
        for file in Path("test-checkpoints-diff-sigma").glob("*"):
            file.unlink()
        Path("test-checkpoints-diff-sigma").rmdir()


def test_different_sigma_rels_with_only_save_diff():
    """Test that different sigma_rel values produce different weights with only_save_diff=True."""
    # Create a simple model
    model = torch.nn.Sequential(
        torch.nn.Linear(512, 512),
        torch.nn.BatchNorm1d(512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 10),
    )
    model.train()

    # Disable gradients for some parameters
    model[0].bias.requires_grad = False
    model[1].weight.requires_grad = False
    model[1].bias.requires_grad = False

    # Print initial model parameters
    print("\nInitial model parameters:")
    for name, param in model.named_parameters():
        print(f"{name}: requires_grad={param.requires_grad}, shape={param.shape}")

    # Create EMA instance with only_save_diff=True
    posthoc_ema = PostHocEMA.from_model(
        model,
        "test-checkpoints-diff-sigma",
        checkpoint_every=5,
        sigma_rels=(0.05, 0.28),
        only_save_diff=True,  # Only save parameters that require gradients
        update_after_step=0,  # Start immediately to match original behavior
    )

    # Do some training to build up EMA weights
    for _ in range(20):
        with torch.no_grad():
            # Only update parameters that require gradients
            for param in model.parameters():
                if param.requires_grad:
                    param.add_(torch.randn_like(param) * 0.1)
        posthoc_ema.update_(model)

    # Get weights for two different sigma_rels
    with posthoc_ema.state_dict(sigma_rel=0.05) as state_dict_1:
        with posthoc_ema.state_dict(sigma_rel=0.4) as state_dict_2:
            # Verify that parameters without gradients are not in state dict
            assert "0.bias" not in state_dict_1
            assert "1.weight" not in state_dict_1
            assert "1.bias" not in state_dict_1
            assert "0.bias" not in state_dict_2
            assert "1.weight" not in state_dict_2
            assert "1.bias" not in state_dict_2

            # Compare weights that are present
            max_diff = 0.0
            mean_diff = 0.0
            num_params = 0

            for key in state_dict_1.keys():
                # Skip batch norm running statistics
                if "running_" in key:
                    continue

                if key in state_dict_2:
                    diff = (state_dict_1[key] - state_dict_2[key]).abs()
                    # Skip integer tensors or convert to float for mean calculation
                    if diff.dtype in [torch.int, torch.int8, torch.int16, torch.int32, torch.int64, torch.long]:
                        if "num_batches_tracked" in key:
                            continue  # Skip entirely
                        diff = diff.float()
                    max_diff = max(max_diff, diff.max().item())
                    mean_diff += diff.mean().item()
                    num_params += 1

            mean_diff /= num_params

            print(f"Max difference between weights (only_save_diff=True): {max_diff}")
            print(f"Mean difference between weights (only_save_diff=True): {mean_diff}")

            # Assert weights are different
            assert (
                max_diff > 1e-4
            ), "Weights should be different for different sigma_rels"

            # Also verify that no weights are identical (except running stats)
            for key in state_dict_1.keys():
                # Skip batch norm running statistics and num_batches_tracked
                if "running_" in key or "num_batches_tracked" in key:
                    continue

                if key in state_dict_2:
                    assert not torch.allclose(
                        state_dict_1[key], state_dict_2[key], rtol=1e-5, atol=1e-5
                    ), f"Weights for {key} should be different"

            # Create test input and verify predictions are different
            test_input = torch.randn(16, 512)
            model.eval()

            # Load first state dict
            model.load_state_dict(state_dict_1, strict=False)
            predictions_1 = model(test_input)

            # Load second state dict
            model.load_state_dict(state_dict_2, strict=False)
            predictions_2 = model(test_input)

            # Compare predictions
            max_pred_diff = (predictions_1 - predictions_2).abs().max().item()
            mean_pred_diff = (predictions_1 - predictions_2).abs().mean().item()

            print(
                f"Max difference between predictions (only_save_diff=True): {max_pred_diff}"
            )
            print(
                f"Mean difference between predictions (only_save_diff=True): {mean_pred_diff}"
            )

            # Assert predictions are different
            assert not torch.allclose(
                predictions_1, predictions_2, rtol=1e-5, atol=1e-5
            ), "Predictions should be different for different sigma_rels"

            assert max_pred_diff > 1e-4, "Predictions should be significantly different"

    # Clean up
    if Path("test-checkpoints-diff-sigma").exists():
        for file in Path("test-checkpoints-diff-sigma").glob("*"):
            file.unlink()
        Path("test-checkpoints-diff-sigma").rmdir()


def test_only_save_diff_doesnt_affect_grad_params():
    """Test that only_save_diff=True doesn't affect parameters that require gradients."""
    # Create a simple model
    model = torch.nn.Sequential(
        torch.nn.Linear(512, 512),
        torch.nn.BatchNorm1d(512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 10),
    )
    model.train()

    # Disable gradients for some parameters
    model[0].bias.requires_grad = False
    model[1].weight.requires_grad = False
    model[1].bias.requires_grad = False

    # Print initial model parameters
    print("\nInitial model parameters:")
    for name, param in model.named_parameters():
        print(f"{name}: requires_grad={param.requires_grad}, shape={param.shape}")

    # Create two EMA instances, one with only_save_diff=True and one with False
    posthoc_ema_with_diff = PostHocEMA.from_model(
        model,
        "test-checkpoints-diff-sigma",
        checkpoint_every=1,  # Checkpoint every update for debugging
        sigma_rels=(0.05, 0.4),
        only_save_diff=True,
        update_after_step=0,  # Start immediately to match original behavior
    )

    posthoc_ema_without_diff = PostHocEMA.from_model(
        model,
        "test-checkpoints-diff-sigma",
        checkpoint_every=1,  # Checkpoint every update for debugging
        sigma_rels=(0.05, 0.4),
        only_save_diff=False,
        update_after_step=0,  # Start immediately to match original behavior
    )

    # Do some training to build up EMA weights
    print("\nTraining updates:")
    for i in range(5):  # Reduced iterations for debugging
        print(f"\nUpdate {i + 1}:")
        with torch.no_grad():
            for name, param in model.named_parameters():
                update = torch.randn_like(param) * 0.1
                param.add_(update)
                print(
                    f"Updated {name} with max change: {update.abs().max().item():.6f}"
                )

        posthoc_ema_with_diff.update_(model)
        posthoc_ema_without_diff.update_(model)

    # Print checkpoint files
    print("\nCheckpoint files:")
    print("With diff:")
    for f in sorted(Path("test-checkpoints-diff-sigma").glob("*.pt")):
        print(f"  {f.name}")
        # Load checkpoint and print its keys
        checkpoint = torch.load(str(f))
        print(
            f"  Keys in checkpoint: {sorted(k for k in checkpoint.keys() if k not in ['initted', 'step'])}"
        )
    print("Without diff:")
    for f in sorted(Path("test-checkpoints-diff-sigma").glob("*.pt")):
        print(f"  {f.name}")
        # Load checkpoint and print its keys
        checkpoint = torch.load(str(f))
        print(
            f"  Keys in checkpoint: {sorted(k for k in checkpoint.keys() if k not in ['initted', 'step'])}"
        )

    # Compare state dicts for both sigma_rel values
    for sigma_rel in [0.05, 0.4]:
        with posthoc_ema_with_diff.state_dict(
            sigma_rel=sigma_rel
        ) as state_dict_with_diff:
            with posthoc_ema_without_diff.state_dict(
                sigma_rel=sigma_rel
            ) as state_dict_without_diff:
                # Get the intersection of keys that exist in both state dicts
                common_keys = set(state_dict_with_diff.keys()) & set(
                    state_dict_without_diff.keys()
                )

                print(f"\nComparing state dicts for sigma_rel={sigma_rel}")
                print(
                    f"Keys in state_dict_with_diff: {sorted(state_dict_with_diff.keys())}"
                )
                print(
                    f"Keys in state_dict_without_diff: {sorted(state_dict_without_diff.keys())}"
                )
                print(f"Common keys: {sorted(common_keys)}")

                # Compare parameters that have requires_grad=True
                for key in common_keys:
                    param_with_diff = state_dict_with_diff[key]
                    param_without_diff = state_dict_without_diff[key]

                    # Check if parameters match exactly
                    diff = (param_with_diff - param_without_diff).abs()
                    # Handle integer tensors
                    if diff.dtype in [torch.int, torch.int8, torch.int16, torch.int32, torch.int64, torch.long]:
                        if "num_batches_tracked" in key:
                            continue  # Skip integer tracking tensors
                        diff = diff.float()
                    max_diff = diff.max().item()
                    mean_diff = diff.mean().item()

                    print(f"\nParameter {key}:")
                    print(f"  Max difference: {max_diff}")
                    print(f"  Mean difference: {mean_diff}")
                    print(f"  Shape: {param_with_diff.shape}")
                    print(f"  Requires grad: {param_with_diff.requires_grad}")
                    print(f"  Device: {param_with_diff.device}")
                    print(f"  Dtype: {param_with_diff.dtype}")

                    # Assert parameters match
                    assert torch.allclose(
                        param_with_diff, param_without_diff, rtol=1e-5, atol=1e-5
                    ), f"Parameter {key} differs between only_save_diff=True and False"

    # Clean up
    for path in ["test-checkpoints-diff-sigma"]:
        if Path(path).exists():
            for file in Path(path).glob("*"):
                file.unlink()
            Path(path).rmdir()
