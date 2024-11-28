"""Tests to verify our implementation matches the reference implementation."""

import shutil
from pathlib import Path

import pytest
import torch
from torch import nn

from posthoc_ema import PostHocEMA as OurPostHocEMA
from tests.lucidrains_posthoc_ema_reference import PostHocEMA as ReferencePostHocEMA


def print_state_comparison(step: int, ref_model: nn.Module, our_model: nn.Module):
    """Print detailed comparison of model states."""
    print(f"\nStep {step}:")
    ref_state = ref_model.state_dict()
    our_state = our_model.state_dict()
    
    # Get common keys (model parameters)
    if "ema_model.weight" in ref_state:
        # Reference implementation includes EMA model state
        ref_keys = {k.replace("ema_model.", ""): k for k in ref_state.keys() 
                   if k.startswith("ema_model.")}
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
                print(f"Index {idx}: Ref={ref_val:.6f}, Our={our_val:.6f}, Diff={ref_val-our_val:.6f}")


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


def test_same_output_as_reference():
    """Test that our implementation produces identical outputs to the reference."""
    # Create a simple model
    net = nn.Linear(512, 512)
    
    # Initialize with same parameters
    sigma_rels = (0.05, 0.28)
    update_every = 10
    checkpoint_every = 10
    
    # Create both implementations
    ref_emas = ReferencePostHocEMA(
        net,
        sigma_rels=sigma_rels,
        update_every=update_every,
        checkpoint_every_num_steps=checkpoint_every,
        checkpoint_folder="./test-checkpoints-ref",
    )
    
    our_emas = OurPostHocEMA(
        net,
        checkpoint_dir="./test-checkpoints-our",
        update_every=update_every,
        checkpoint_every=checkpoint_every,
        sigma_rels=sigma_rels,
    )

    # Train both with identical updates
    torch.manual_seed(42)  # For reproducibility
    net.train()

    print("\nInitial state:")
    print_state_comparison(
        -1, 
        ref_emas.ema_models[0].ema_model,
        our_emas.ema_models[0].ema_model
    )

    for step in range(100):
        # Apply identical mutations to network
        with torch.no_grad():
            net.weight.copy_(torch.randn_like(net.weight))
            net.bias.copy_(torch.randn_like(net.bias))

        # Debug: Print state before update
        if step % 10 == 0:
            print(f"\nBefore update {step}:")
            print(f"Ref beta: {ref_emas.ema_models[0].beta:.6f}")
            print(f"Our beta: {our_emas.ema_models[0].beta:.6f}")

        # Update both EMA wrappers
        ref_emas.update()
        our_emas.update(net)

        # Debug: Print state after update
        if step % 10 == 0:
            print_state_comparison(
                step,
                ref_emas.ema_models[0].ema_model,
                our_emas.ema_models[0].ema_model
            )

    # Synthesize EMA models with same parameters
    target_sigma = 0.15
    ref_synth = ref_emas.synthesize_ema_model(sigma_rel=target_sigma)
    
    with our_emas.model(net, sigma_rel=target_sigma) as our_synth:
        # Test with same input
        data = torch.randn(1, 512)
        ref_output = ref_synth(data)
        our_output = our_synth(data)

        # Debug: Print synthesis details
        print("\nSynthesis details:")
        print(f"Target sigma_rel: {target_sigma}")
        print(f"Target gamma: {ref_emas.gammas[0]:.6f}")
        
        print("\nFinal model states:")
        print_state_comparison(step, ref_synth, our_synth)

        # Verify outputs match
        output_diff = (ref_output - our_output).abs()
        print("\nOutput comparison:")
        print(f"Max output diff: {output_diff.max():.6f}")
        print(f"Mean output diff: {output_diff.mean():.6f}")
        
        assert torch.allclose(
            ref_output, our_output, rtol=1e-5, atol=1e-5
        ), "Output from our implementation doesn't match reference"

        # Verify model parameters match
        for (ref_name, ref_param), (our_name, our_param) in zip(
            ref_synth.named_parameters(), our_synth.named_parameters()
        ):
            assert ref_name == our_name, \
                f"Parameter names don't match: {ref_name} vs {our_name}"
            assert torch.allclose(
                ref_param, our_param, rtol=1e-5, atol=1e-5
            ), f"Parameters don't match for {ref_name}"
