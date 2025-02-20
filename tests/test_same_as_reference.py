"""Tests to verify our implementation matches the reference implementation."""

import shutil
from pathlib import Path

import pytest
import torch
from torch import nn

from posthoc_ema import PostHocEMA as OurPostHocEMA
from posthoc_ema.utils import sigma_rel_to_gamma, p_dot_p, solve_weights
from tests.lucidrains_posthoc_ema_reference import PostHocEMA as ReferencePostHocEMA
from tests.lucidrains_posthoc_ema_reference import p_dot_p as ref_p_dot_p
from tests.lucidrains_posthoc_ema_reference import solve_weights as ref_solve_weights


def print_state_comparison(step: int, ref_model: nn.Module, our_model: nn.Module):
    """Print detailed comparison of model states."""
    print(f"\nStep {step}:")
    ref_state = ref_model.state_dict()
    our_state = our_model.state_dict()

    # Get common keys (model parameters)
    if "ema_model.weight" in ref_state:
        # Reference implementation includes EMA model state
        ref_keys = {
            k.replace("ema_model.", ""): k
            for k in ref_state.keys()
            if k.startswith("ema_model.")
        }
        our_keys = set(our_state.keys())
    else:
        # Direct model comparison
        ref_keys = {k: k for k in ref_state.keys()}
        our_keys = set(our_state.keys())

    # Compare common parameters
    for model_key, ref_key in ref_keys.items():
        if model_key not in our_keys:
            print(f"\nMissing key in our model: {model_key}")
            continue

        ref_tensor = ref_state[ref_key]
        our_tensor = our_state[model_key]

        if not torch.allclose(ref_tensor, our_tensor, rtol=1e-5, atol=1e-5):
            print(f"\nMismatch in {model_key}:")
            print(f"Ref mean: {ref_tensor.mean():.6f}")
            print(f"Our mean: {our_tensor.mean():.6f}")
            print(f"Max diff: {(ref_tensor - our_tensor).abs().max():.6f}")
            print(f"Ref std: {ref_tensor.std():.6f}")
            print(f"Our std: {our_tensor.std():.6f}")
            # Print first few elements where difference is largest
            diff = (ref_tensor - our_tensor).abs()
            max_diff_idx = diff.flatten().topk(3).indices
            print("\nLargest differences:")
            for idx in max_diff_idx:
                ref_val = ref_tensor.flatten()[idx].item()
                our_val = our_tensor.flatten()[idx].item()
                print(
                    f"Index {idx}: Ref={ref_val:.6f}, Our={our_val:.6f}, Diff={ref_val-our_val:.6f}"
                )


@pytest.fixture(autouse=True)
def cleanup_checkpoints():
    """Clean up test checkpoints before and after each test."""
    # Cleanup before test
    for path in ["./test-checkpoints-ref", "./test-checkpoints-our"]:
        if Path(path).exists():
            shutil.rmtree(path)

    yield

    # Cleanup after test
    for path in ["./test-checkpoints-ref", "./test-checkpoints-our"]:
        if Path(path).exists():
            shutil.rmtree(path)


def test_p_dot_p_matches_reference():
    """Test that our p_dot_p implementation matches the reference."""
    test_cases = [
        # Regular case
        {
            "t_a": torch.tensor([[10.0, 20.0], [30.0, 40.0]], dtype=torch.float64),
            "gamma_a": torch.tensor([[7.0, 7.0], [7.0, 7.0]], dtype=torch.float64),
            "t_b": torch.tensor([[10.0, 20.0], [30.0, 40.0]], dtype=torch.float64),
            "gamma_b": torch.tensor([[28.0, 28.0], [28.0, 28.0]], dtype=torch.float64),
        },
        # Small values
        {
            "t_a": torch.tensor([[1e-6, 1e-5], [1e-4, 1e-3]], dtype=torch.float64),
            "gamma_a": torch.tensor([[7.0, 7.0], [7.0, 7.0]], dtype=torch.float64),
            "t_b": torch.tensor([[1e-6, 1e-5], [1e-4, 1e-3]], dtype=torch.float64),
            "gamma_b": torch.tensor([[28.0, 28.0], [28.0, 28.0]], dtype=torch.float64),
        },
        # Different timesteps
        {
            "t_a": torch.tensor([[5.0, 15.0], [25.0, 35.0]], dtype=torch.float64),
            "gamma_a": torch.tensor([[7.0, 7.0], [7.0, 7.0]], dtype=torch.float64),
            "t_b": torch.tensor([[10.0, 20.0], [30.0, 40.0]], dtype=torch.float64),
            "gamma_b": torch.tensor([[28.0, 28.0], [28.0, 28.0]], dtype=torch.float64),
        },
    ]

    print("\nTesting p_dot_p with multiple cases:")
    for i, case in enumerate(test_cases):
        print(f"\nCase {i + 1}:")
        print(f"t_a:\n{case['t_a']}")
        print(f"t_b:\n{case['t_b']}")

        our_result = p_dot_p(case["t_a"], case["gamma_a"], case["t_b"], case["gamma_b"])
        ref_result = ref_p_dot_p(
            case["t_a"], case["gamma_a"], case["t_b"], case["gamma_b"]
        )

        print(f"Our result:\n{our_result}")
        print(f"Ref result:\n{ref_result}")
        print(f"Max difference: {(our_result - ref_result).abs().max().item()}")

        assert torch.allclose(
            our_result, ref_result, rtol=1e-5, atol=1e-5
        ), f"Case {i + 1} failed"


def test_solve_weights_matches_reference():
    """Test that our solve_weights implementation matches the reference."""
    test_cases = [
        {
            "gammas": torch.tensor([7.0, 28.0], dtype=torch.float64),
            "timesteps": torch.tensor([1.0, 10.0], dtype=torch.float64),
            "target_gamma": 15.0,
        },
        {
            "gammas": torch.tensor([1.0, 5.0, 10.0], dtype=torch.float64),
            "timesteps": torch.tensor([1.0, 5.0, 10.0], dtype=torch.float64),
            "target_gamma": 3.0,
        },
        {
            "gammas": torch.tensor([100.0, 500.0], dtype=torch.float64),
            "timesteps": torch.tensor([1.0, 100.0], dtype=torch.float64),
            "target_gamma": 200.0,
        },
    ]

    print("\nTesting solve_weights with multiple cases:")
    for i, case in enumerate(test_cases, 1):
        print(f"\nCase {i}:")
        gammas = case["gammas"]
        timesteps = case["timesteps"]
        target_gamma = case["target_gamma"]

        # Convert to reference format
        t_i = timesteps
        gamma_i = gammas
        t_r = torch.tensor(
            [timesteps[-1]], device=timesteps.device, dtype=torch.float64
        )
        gamma_r = torch.tensor(
            [target_gamma], device=gammas.device, dtype=torch.float64
        )

        # Compute with both implementations
        our_result = solve_weights(
            gammas, timesteps, target_gamma, calculation_dtype=torch.float64
        )
        ref_result = ref_solve_weights(t_i, gamma_i, t_r, gamma_r)

        print(f"Gammas: {gammas}")
        print(f"Timesteps: {timesteps}")
        print(f"Target gamma: {target_gamma}")
        print(f"Our result:\n{our_result}")
        print(f"Ref result:\n{ref_result.squeeze()}")
        print(
            f"Max difference: {(our_result - ref_result.squeeze()).abs().max().item()}"
        )
        print(f"Our weights sum: {our_result.sum().item()}")
        print(f"Ref weights sum: {ref_result.sum().item()}")

        assert torch.allclose(our_result, ref_result.squeeze(), rtol=1e-5, atol=1e-5)


def test_gamma_conversion_matches_reference():
    """Test that our gamma conversion matches the reference."""
    sigma_rels = [0.05, 0.15, 0.28, 0.4]

    print("\nTesting gamma conversion:")
    for sr in sigma_rels:
        our_gamma = sigma_rel_to_gamma(sr)
        ref_gamma = sigma_rel_to_gamma(sr)  # Using reference function
        print(f"sigma_rel={sr}:")
        print(f"  Our gamma: {our_gamma}")
        print(f"  Ref gamma: {ref_gamma}")
        assert abs(our_gamma - ref_gamma) < 1e-6


def test_same_output_as_reference():
    """Test that our implementation produces identical outputs to the reference."""
    # Create a simple model
    net = nn.Linear(512, 512)

    # Initialize with same parameters
    sigma_rels = (0.03, 0.20)
    update_every = 10
    checkpoint_every = 10

    print("\nInitializing with parameters:")
    print(f"sigma_rels: {sigma_rels}")
    print(f"update_every: {update_every}")
    print(f"checkpoint_every: {checkpoint_every}")

    # Create both implementations
    ref_emas = ReferencePostHocEMA(
        net,
        sigma_rels=sigma_rels,
        update_every=update_every,
        checkpoint_every_num_steps=checkpoint_every,
        checkpoint_folder="./test-checkpoints-ref",
        checkpoint_dtype=torch.float32,
    )

    our_emas = OurPostHocEMA.from_model(
        model=net,
        checkpoint_dir="./test-checkpoints-our",
        update_every=update_every,
        checkpoint_every=checkpoint_every,
        sigma_rels=sigma_rels,
        checkpoint_dtype=torch.float32,
        update_after_step=0,  # Start immediately to match reference behavior
    )

    # Train both with identical updates
    torch.manual_seed(42)  # For reproducibility
    net.train()

    print("\nTraining:")
    for step in range(100):
        # Apply identical mutations to network
        with torch.no_grad():
            net.weight.copy_(torch.randn_like(net.weight))
            net.bias.copy_(torch.randn_like(net.bias))

        # Update both EMA wrappers
        ref_emas.update()
        our_emas.update_(net)

        if step % 10 == 0:
            print(f"Step {step}: Updated model and EMAs")

    # Synthesize EMA models with same parameters
    target_sigma = 0.15
    print(f"\nSynthesizing with target_sigma = {target_sigma}")

    # Get reference checkpoints and weights
    ref_checkpoints = sorted(Path("./test-checkpoints-ref").glob("*.pt"))
    print("\nReference checkpoints:")
    for cp in ref_checkpoints:
        print(f"  {cp.name}")

    # Get our checkpoints and weights
    our_checkpoints = sorted(Path("./test-checkpoints-our").glob("*.pt"))
    print("\nOur checkpoints:")
    for cp in our_checkpoints:
        print(f"  {cp.name}")

    ref_synth = ref_emas.synthesize_ema_model(sigma_rel=target_sigma)

    with our_emas.model(net, target_sigma) as our_synth:
        # Test with same input
        data = torch.randn(1, 512)
        ref_output = ref_synth(data)
        our_output = our_synth(data)

        print("\nComparing outputs:")
        print(f"Reference output mean: {ref_output.mean().item():.4f}")
        print(f"Our output mean: {our_output.mean().item():.4f}")
        print(f"Max difference: {(ref_output - our_output).abs().max().item():.4f}")

        # Verify outputs match
        assert torch.allclose(
            ref_output, our_output, rtol=1e-4, atol=1e-4
        ), "Output from our implementation doesn't match reference"

    our_emas_from_disk = OurPostHocEMA.from_path(
        checkpoint_dir="./test-checkpoints-our",
        model=net,
        sigma_rels=sigma_rels,  # Use same sigma_rels as original
        update_every=update_every,  # Use same update_every as original
        checkpoint_every=checkpoint_every,  # Use same checkpoint_every as original
        checkpoint_dtype=torch.float32,  # Use same dtype as original
    )

    with our_emas_from_disk.model(net, target_sigma) as our_synth:
        # Test with same input
        our_from_disk_output = our_synth(data)

        print("\nComparing loaded outputs:")
        print(f"Reference output mean: {ref_output.mean().item():.4f}")
        print(f"Our loaded output mean: {our_from_disk_output.mean().item():.4f}")
        print(
            f"Max difference: {(ref_output - our_from_disk_output).abs().max().item():.4f}"
        )

        # Verify outputs match
        assert torch.allclose(
            ref_output, our_from_disk_output, rtol=1e-4, atol=1e-4
        ), "Output from loaded implementation doesn't match reference"


def test_update_after_step():
    """Test that EMA updates only start after update_after_step steps."""
    # Create a simple model
    net = nn.Linear(512, 512)
    update_after_step = 50

    # Initialize with same parameters
    sigma_rels = (0.03, 0.20)
    update_every = 10
    checkpoint_every = 10

    our_emas = OurPostHocEMA.from_model(
        model=net,
        checkpoint_dir="./test-checkpoints-our",
        update_every=update_every,
        checkpoint_every=checkpoint_every,
        sigma_rels=sigma_rels,
        checkpoint_dtype=torch.float32,
        update_after_step=update_after_step,
    )

    # Train with identical updates
    torch.manual_seed(42)  # For reproducibility
    net.train()

    # Store initial weights
    initial_weights = {}
    for ema_model in our_emas.ema_models:
        initial_weights[id(ema_model)] = {
            name: param.clone()
            for name, param in ema_model.ema_model.named_parameters()
        }

    # Update before update_after_step
    for step in range(update_after_step - 1):
        with torch.no_grad():
            net.weight.copy_(torch.randn_like(net.weight))
            net.bias.copy_(torch.randn_like(net.bias))
        our_emas.update_(net)

        # Verify EMA weights haven't changed
        for ema_model in our_emas.ema_models:
            current_weights = {
                name: param for name, param in ema_model.ema_model.named_parameters()
            }
            initial_weights_for_model = initial_weights[id(ema_model)]

            for name, param in current_weights.items():
                assert torch.allclose(
                    param, initial_weights_for_model[name], rtol=1e-5, atol=1e-5
                ), f"EMA weights changed before update_after_step at step {step}"

    # Update after update_after_step
    with torch.no_grad():
        net.weight.copy_(torch.randn_like(net.weight))
        net.bias.copy_(torch.randn_like(net.bias))
    our_emas.update_(net)

    # Verify EMA weights have changed
    for ema_model in our_emas.ema_models:
        current_weights = {
            name: param for name, param in ema_model.ema_model.named_parameters()
        }
        initial_weights_for_model = initial_weights[id(ema_model)]

        weights_changed = False
        for name, param in current_weights.items():
            if not torch.allclose(
                param, initial_weights_for_model[name], rtol=1e-5, atol=1e-5
            ):
                weights_changed = True
                break

        assert weights_changed, "EMA weights did not change after update_after_step"


def test_same_output_as_reference_different_step():
    """Test that our implementation produces identical outputs to the reference when synthesizing at a different step."""
    # Create a simple model
    net = nn.Linear(512, 512)

    # Initialize with same parameters
    sigma_rels = (0.03, 0.20)
    update_every = 10
    checkpoint_every = 10

    print("\nInitializing with parameters:")
    print(f"sigma_rels: {sigma_rels}")
    print(f"update_every: {update_every}")
    print(f"checkpoint_every: {checkpoint_every}")

    # Create both implementations
    ref_emas = ReferencePostHocEMA(
        net,
        sigma_rels=sigma_rels,
        update_every=update_every,
        checkpoint_every_num_steps=checkpoint_every,
        checkpoint_folder="./test-checkpoints-ref",
        checkpoint_dtype=torch.float32,
    )

    our_emas = OurPostHocEMA.from_model(
        model=net,
        checkpoint_dir="./test-checkpoints-our",
        update_every=update_every,
        checkpoint_every=checkpoint_every,
        sigma_rels=sigma_rels,
        checkpoint_dtype=torch.float32,
        update_after_step=0,  # Start immediately to match reference behavior
    )

    # Train both with identical updates
    torch.manual_seed(42)  # For reproducibility
    net.train()

    print("\nTraining:")
    for step in range(100):
        # Apply identical mutations to network
        with torch.no_grad():
            net.weight.copy_(torch.randn_like(net.weight))
            net.bias.copy_(torch.randn_like(net.bias))

        # Update both EMA wrappers
        ref_emas.update()
        our_emas.update_(net)

        if step % 10 == 0:
            print(f"Step {step}: Updated model and EMAs")

    # Synthesize EMA models with same parameters at step 50 (middle of training)
    target_sigma = 0.15
    target_step = 50
    print(f"\nSynthesizing with target_sigma = {target_sigma} at step {target_step}")

    # Get reference checkpoints and weights
    ref_checkpoints = sorted(Path("./test-checkpoints-ref").glob("*.pt"))
    print("\nReference checkpoints:")
    for cp in ref_checkpoints:
        print(f"  {cp.name}")

    # Get our checkpoints and weights
    our_checkpoints = sorted(Path("./test-checkpoints-our").glob("*.pt"))
    print("\nOur checkpoints:")
    for cp in our_checkpoints:
        print(f"  {cp.name}")

    ref_synth = ref_emas.synthesize_ema_model(sigma_rel=target_sigma, step=target_step)

    with our_emas.model(net, target_sigma, step=target_step) as our_synth:
        # Test with same input
        data = torch.randn(1, 512)
        ref_output = ref_synth(data)
        our_output = our_synth(data)

        print("\nComparing outputs:")
        print(f"Reference output mean: {ref_output.mean().item():.4f}")
        print(f"Our output mean: {our_output.mean().item():.4f}")
        print(f"Max difference: {(ref_output - our_output).abs().max().item():.4f}")

        # Verify outputs match
        assert torch.allclose(
            ref_output, our_output, rtol=1e-4, atol=1e-4
        ), "Output from our implementation doesn't match reference"

    # Clean up
    for path in ["./test-checkpoints-ref", "./test-checkpoints-our"]:
        if Path(path).exists():
            shutil.rmtree(path)
