"""Tests to verify our implementation matches the reference implementation."""

import torch
from torch import nn

from posthoc_ema import PostHocEMA as OurPostHocEMA
from tests.lucidrains_posthoc_ema_reference import PostHocEMA as ReferencePostHocEMA


def test_same_output_as_reference():
    """Test that our implementation produces identical outputs to the reference."""
    # Create a simple model
    net = nn.Linear(512, 512)

    # Initialize with same parameters
    sigma_rels = (0.05, 0.28)
    update_every = 10
    checkpoint_every = 10
    checkpoint_folder = "./test-checkpoints"

    # Create both implementations
    ref_emas = ReferencePostHocEMA(
        net,
        sigma_rels=sigma_rels,
        update_every=update_every,
        checkpoint_every_num_steps=checkpoint_every,
        checkpoint_folder=f"{checkpoint_folder}-ref",
    )

    our_emas = OurPostHocEMA(
        net,
        sigma_rels=sigma_rels,
        update_every=update_every,
        checkpoint_every_num_steps=checkpoint_every,
        checkpoint_folder=f"{checkpoint_folder}-our",
    )

    # Train both with identical updates
    torch.manual_seed(42)  # For reproducibility
    net.train()

    for _ in range(100):
        # Apply identical mutations to network
        with torch.no_grad():
            net.weight.copy_(torch.randn_like(net.weight))
            net.bias.copy_(torch.randn_like(net.bias))

        # Update both EMA wrappers
        ref_emas.update()
        our_emas.update()

    # Synthesize EMA models with same parameters
    target_sigma = 0.15
    ref_synth = ref_emas.synthesize_ema_model(sigma_rel=target_sigma)
    our_synth = our_emas.synthesize_ema_model(sigma_rel=target_sigma)

    # Test with same input
    data = torch.randn(1, 512)
    ref_output = ref_synth(data)
    our_output = our_synth(data)

    # Verify outputs match
    assert torch.allclose(
        ref_output, our_output, rtol=1e-5, atol=1e-5
    ), "Output from our implementation doesn't match reference"

    # Verify model parameters match
    for (ref_name, ref_param), (our_name, our_param) in zip(
        ref_synth.named_parameters(), our_synth.named_parameters()
    ):
        assert (
            ref_name == our_name
        ), f"Parameter names don't match: {ref_name} vs {our_name}"
        assert torch.allclose(
            ref_param, our_param, rtol=1e-5, atol=1e-5
        ), f"Parameters don't match for {ref_name}"
