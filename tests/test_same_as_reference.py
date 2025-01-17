"""Tests to verify our implementation matches the reference implementation."""

import shutil
from pathlib import Path

import pytest
import torch
from torch import nn

from posthoc_ema import PostHocEMA as OurPostHocEMA
from posthoc_ema.utils import sigma_rel_to_gamma
from tests.lucidrains_posthoc_ema_reference import PostHocEMA as ReferencePostHocEMA


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
        checkpoint_dtype=torch.float32,
    )

    our_emas = OurPostHocEMA.from_model(
        model=net,
        checkpoint_dir="./test-checkpoints-our",
        update_every=update_every,
        checkpoint_every=checkpoint_every,
        sigma_rels=sigma_rels,
        checkpoint_dtype=torch.float32,
    )

    # Train both with identical updates
    torch.manual_seed(42)  # For reproducibility
    net.train()

    print("\nInitial state:")
    print_state_comparison(
        -1, ref_emas.ema_models[0].ema_model, our_emas.ema_models[0].ema_model
    )

    for step in range(100):
        # Apply identical mutations to network
        with torch.no_grad():
            net.weight.copy_(torch.randn_like(net.weight))
            net.bias.copy_(torch.randn_like(net.bias))

        # Debug: Print state before update
        if step % 10 == 0:
            print(f"\nStep {step} before update:")
            print("\nOnline model state:")
            print(f"Weight mean: {net.weight.mean():.6f}")
            print(f"Weight std: {net.weight.std():.6f}")
            print(f"Bias mean: {net.bias.mean():.6f}")
            print(f"Bias std: {net.bias.std():.6f}")

        # Update both EMA wrappers
        ref_emas.update()
        our_emas.update_(net)

        # Debug: Print state after update
        if step % 10 == 0:
            print("\nAfter update:")
            print_state_comparison(
                step, ref_emas.ema_models[0].ema_model, our_emas.ema_models[0].ema_model
            )

            # Print checkpoint info if created
            if step % checkpoint_every == 0:
                print("\nCheckpoint created:")
                ref_files = sorted(Path("./test-checkpoints-ref").glob("*.pt"))
                our_files = sorted(Path("./test-checkpoints-our").glob("*.pt"))
                print(f"Ref checkpoints: {len(ref_files)}")
                print(f"Our checkpoints: {len(our_files)}")

    # Print final checkpoint comparison
    print("\nFinal checkpoint comparison:")
    ref_files = sorted(Path("./test-checkpoints-ref").glob("*.pt"))
    our_files = sorted(Path("./test-checkpoints-our").glob("*.pt"))

    for ref_file, our_file in zip(ref_files, our_files):
        ref_state = torch.load(ref_file, weights_only=True)
        our_state = torch.load(our_file, weights_only=True)
        print(f"\nComparing {ref_file.name} vs {our_file.name}:")

        # Map reference keys to our keys
        ref_model_keys = {
            k: k.replace("ema_model.", "")
            for k in ref_state.keys()
            if k.startswith("ema_model.")
        }

        # Compare model parameters
        for ref_key, our_key in ref_model_keys.items():
            if our_key not in our_state:
                print(f"Missing key in our checkpoint: {our_key}")
                continue

            ref_tensor = ref_state[ref_key]
            our_tensor = our_state[our_key]
            max_diff = (ref_tensor - our_tensor).abs().max().item()
            print(f"{ref_key} vs {our_key}: max diff = {max_diff:.6f}")

            # Print more details if difference is large
            if max_diff > 1e-5:
                print(f"  Ref mean: {ref_tensor.mean():.6f}")
                print(f"  Our mean: {our_tensor.mean():.6f}")
                print(f"  Ref std: {ref_tensor.std():.6f}")
                print(f"  Our std: {our_tensor.std():.6f}")

    # Synthesize EMA models with same parameters
    target_sigma = 0.15
    ref_synth = ref_emas.synthesize_ema_model(sigma_rel=target_sigma)

    with our_emas.model(net, sigma_rel=target_sigma) as our_synth:
        # Debug: Print synthesis details
        print("\nSynthesis details:")
        print(f"Target sigma_rel: {target_sigma}")
        print(f"Target gamma: {sigma_rel_to_gamma(target_sigma):.6f}")

        # Get checkpoint info for weight solving
        device = torch.device("cpu")
        gamma = sigma_rel_to_gamma(target_sigma)

        # Collect checkpoint info
        gammas = []
        timesteps = []
        for idx in range(len(our_emas.ema_models)):
            checkpoint_files = sorted(
                Path("./test-checkpoints-our").glob(f"{idx}.*.pt"),
                key=lambda p: int(p.stem.split(".")[1]),
            )
            for file in checkpoint_files:
                _, timestep = map(int, file.stem.split("."))
                gammas.append(our_emas.gammas[idx])
                timesteps.append(timestep)

        # Print checkpoint info
        print("\nCheckpoint info:")
        print(f"Number of checkpoints: {len(timesteps)}")
        print(f"Timesteps: {timesteps}")
        print(f"Gammas: {gammas}")

        # Test with same input
        data = torch.randn(1, 512)
        ref_output = ref_synth(data)
        our_output = our_synth(data)

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
        ref_params = {
            name.replace("ema_model.", ""): param 
            for name, param in ref_synth.named_parameters()
        }
        our_params = dict(our_synth.named_parameters())

        for name in ref_params:
            assert name in our_params, f"Missing parameter in our model: {name}"
            assert torch.allclose(
                ref_params[name], our_params[name], rtol=1e-5, atol=1e-5
            ), f"Parameters don't match for {name}"
