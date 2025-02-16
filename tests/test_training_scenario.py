from pathlib import Path

import torch
from torch import nn

from posthoc_ema import PostHocEMA
from posthoc_ema.karras_ema import KarrasEMA


def test_training_scenario(tmp_path: Path):
    """Test PostHocEMA in a simulated training scenario.

    This test simulates:
    1. Regular model updates during training
    2. Updates for both Karras EMA and post-hoc EMA
    3. Comparing synthesized post-hoc EMA with Karras EMA
    """
    # Set up model
    model = nn.Linear(10, 5)

    # Initialize KarrasEMA
    vanilla_ema = KarrasEMA(
        model=model,
        sigma_rel=0.15,  # Match post-hoc EMA
        update_every=1,
    )

    # Initialize PostHocEMA
    posthoc_ema = PostHocEMA.from_model(
        model=model,
        checkpoint_dir=tmp_path,
        max_checkpoints=100,
        sigma_rels=(0.15,),  # Match Karras EMA
        update_every=1,
        checkpoint_every=1,  # Create checkpoints every step
        update_after_step=0,  # Start immediately to match original behavior
    )

    # Generate random input for testing
    x = torch.randn(32, 10)
    initial_pred = model(x)

    # Training loop
    for epoch in range(2):
        for step in range(100):  # Run for even more steps
            # Update model weights randomly
            for param in model.parameters():
                param.data += 0.01 * torch.randn_like(param)

            # Update both EMAs
            vanilla_ema.update()
            posthoc_ema.update_(model)

            # Every 5 steps, compare vanilla EMA with synthesized post-hoc EMA
            # But only after we have at least 20 checkpoints
            if step > 0 and step % 5 == 0 and step >= 20:
                # Get predictions from Karras EMA
                with torch.no_grad():
                    vanilla_pred = vanilla_ema.ema_model(x)

                # Get predictions from synthesized post-hoc EMA
                with posthoc_ema.model(model, sigma_rel=0.15) as ema_model:
                    posthoc_pred = ema_model(x)

                # Compare predictions with higher tolerance
                assert torch.allclose(
                    vanilla_pred, posthoc_pred, rtol=5e-2, atol=5e-2
                ), f"EMA predictions differ at step {step}: max diff = {(vanilla_pred - posthoc_pred).abs().max().item()}"

                # Print max difference for inspection
                print(
                    f"Step {step}: max diff = {(vanilla_pred - posthoc_pred).abs().max().item()}"
                )
