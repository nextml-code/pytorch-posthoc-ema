"""
Test to demonstrate how different sigma_rel values affect a model with a single parameter.

This file contains practical tests that show the behavior of PostHocEMA with different
sigma_rel values. The tests demonstrate that:

1. Different sigma_rel values produce different EMA results
2. In PostHocEMA, smaller sigma_rel values (e.g., 0.05) result in EMA values that are
   closer to recent model values, while larger sigma_rel values (e.g., 0.27) result in
   EMA values that are closer to older model values.
3. The relationship between sigma_rel and EMA behavior in PostHocEMA is:
   - sigma_rel ≈ 0.05: More weight to recent values
   - sigma_rel ≈ 0.15: Balanced weighting
   - sigma_rel ≈ 0.27: More weight to older values

Note: This behavior might seem counterintuitive when compared to the relationship between
sigma_rel and beta (EMA decay rate) described in the README:
- Small sigma_rel (e.g., 0.01) corresponds to high beta (e.g., 0.9898) = slow decay
- Large sigma_rel (e.g., 0.27) corresponds to low beta (e.g., 0.2606) = fast decay

The difference is because PostHocEMA is synthesizing weights based on the entire history
of checkpoints, not just applying a simple EMA formula.
"""

import torch
import pytest
import shutil
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from posthoc_ema import PostHocEMA


class SingleParamModel(torch.nn.Module):
    """A model with a single parameter for testing EMA behavior."""

    def __init__(self, initial_value: float = 0.0):
        super().__init__()
        self.param = torch.nn.Parameter(
            torch.tensor([initial_value], dtype=torch.float32)
        )

    def forward(self, x):
        return x * self.param


@pytest.fixture(autouse=True)
def cleanup_checkpoints():
    """Clean up test checkpoints before and after each test."""
    # Cleanup before test
    for path in ["./test-single-param-ema"]:
        if Path(path).exists():
            shutil.rmtree(path)

    yield

    # Cleanup after test
    for path in ["./test-single-param-ema"]:
        if Path(path).exists():
            shutil.rmtree(path)


def test_single_parameter_ema_behavior():
    """
    Test that demonstrates how different sigma_rel values affect a model with a single parameter.

    This test:
    1. Creates a model with a single parameter initialized to 0
    2. Gradually updates the parameter to 1 over 5000 steps
    3. Checks that different sigma_rel values produce different EMA values
    4. Verifies that in PostHocEMA, smaller sigma_rel values result in EMA values
       that are closer to recent model values (closer to 1 in this test)
    """
    # Create a model with a single parameter
    model = SingleParamModel(initial_value=0.0)

    # Create EMA instance with multiple sigma_rel values
    posthoc_ema = PostHocEMA.from_model(
        model,
        "test-single-param-ema",
        checkpoint_every=100,  # Save checkpoints more frequently
        sigma_rels=(0.05, 0.15, 0.27),  # Multiple sigma_rel values
        update_after_step=0,  # Start immediately
    )

    # Number of steps to update from 0 to 1
    num_steps = 5000

    # Track parameter values at different steps
    step_records = []
    param_records = []

    # Gradually update the parameter from 0 to 1
    for step in range(num_steps):
        # Linear interpolation from 0 to 1
        target_value = step / (num_steps - 1)

        with torch.no_grad():
            model.param.copy_(torch.tensor([target_value], dtype=torch.float32))

        posthoc_ema.update_(model)

        # Record values at specific steps
        if step % 500 == 0 or step == num_steps - 1:
            step_records.append(step)
            param_records.append(model.param.item())

    # Print the final model parameter value (should be close to 1)
    print(f"\nFinal model parameter value: {model.param.item()}")

    # Test different sigma_rel values
    sigma_rels = [0.05, 0.15, 0.27]
    ema_values = {}

    for sigma_rel in sigma_rels:
        with posthoc_ema.state_dict(sigma_rel=sigma_rel) as state_dict:
            ema_values[sigma_rel] = state_dict["param"].item()
            print(f"EMA value with sigma_rel={sigma_rel}: {ema_values[sigma_rel]}")

    # Verify that different sigma_rel values produce different results
    assert (
        ema_values[0.05] != ema_values[0.15]
    ), "Different sigma_rel values should produce different results"
    assert (
        ema_values[0.15] != ema_values[0.27]
    ), "Different sigma_rel values should produce different results"

    # Verify that smaller sigma_rel values result in EMA values closer to recent values
    # In PostHocEMA, smaller sigma_rel values give more weight to recent checkpoints
    # Since our parameter is increasing from 0 to 1, recent values are closer to 1
    assert (
        ema_values[0.05] > ema_values[0.15] > ema_values[0.27]
    ), "In PostHocEMA, smaller sigma_rel values should result in EMA values closer to recent values"

    # Verify that all EMA values are less than the final model parameter (which is 1)
    # This is expected because EMA is a weighted average of past values, which were all < 1
    for sigma_rel, value in ema_values.items():
        assert (
            value < 1.0
        ), f"EMA value with sigma_rel={sigma_rel} should be less than 1.0"

    # Print the differences between EMA values and the final model parameter
    # These differences show how much each sigma_rel setting weights older vs. newer values
    for sigma_rel, value in ema_values.items():
        print(f"Difference from final value (sigma_rel={sigma_rel}): {1.0 - value}")


def test_single_parameter_ema_visualization():
    """
    Test that visualizes how different sigma_rel values affect a model with a single parameter.

    This test:
    1. Creates a model with a single parameter initialized to 0
    2. Gradually updates the parameter to 1 over 5000 steps
    3. Records EMA values at regular intervals for different sigma_rel values
    4. Plots the results to visualize the effect of different sigma_rel values
       (in PostHocEMA, smaller sigma_rel values give more weight to recent values)
    """
    # Create a model with a single parameter
    model = SingleParamModel(initial_value=0.0)

    # Create EMA instance with multiple sigma_rel values
    posthoc_ema = PostHocEMA.from_model(
        model,
        "test-single-param-ema",
        checkpoint_every=100,  # Save checkpoints more frequently
        sigma_rels=(0.05, 0.15, 0.27),  # Multiple sigma_rel values
        update_after_step=0,  # Start immediately
    )

    # Number of steps to update from 0 to 1
    num_steps = 5000

    # Record points for visualization
    # Only record every 500 steps to reduce test time
    record_steps = list(range(0, num_steps, 500)) + [num_steps - 1]

    # Track parameter values at different steps
    step_records = []
    param_records = []
    ema_records = {0.05: [], 0.15: [], 0.27: []}

    # Gradually update the parameter from 0 to 1
    for step in range(num_steps):
        # Linear interpolation from 0 to 1
        target_value = step / (num_steps - 1)

        with torch.no_grad():
            model.param.copy_(torch.tensor([target_value], dtype=torch.float32))

        posthoc_ema.update_(model)

        # Only try to record EMA values after we've created some checkpoints
        # (after at least checkpoint_every steps)
        if step in record_steps and step >= 100:  # First checkpoint created at step 100
            step_records.append(step)
            param_records.append(model.param.item())

            # Record EMA values for different sigma_rel values
            for sigma_rel in ema_records.keys():
                with posthoc_ema.state_dict(sigma_rel=sigma_rel) as state_dict:
                    ema_records[sigma_rel].append(state_dict["param"].item())

    # Print final values
    print(f"\nFinal model parameter value: {param_records[-1]}")
    for sigma_rel, values in ema_records.items():
        print(f"Final EMA value with sigma_rel={sigma_rel}: {values[-1]}")

    # Verify that smaller sigma_rel values result in faster adaptation
    assert (
        ema_records[0.05][-1] > ema_records[0.15][-1] > ema_records[0.27][-1]
    ), "Smaller sigma_rel values should result in faster adaptation (values closer to 1)"

    # Skip the actual plotting in automated tests
    # We'll just skip plotting in automated tests to avoid dependencies
    # Uncomment this section to generate plots when running manually
    """
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(step_records, param_records, "k-", label="Model Parameter")

    for sigma_rel, values in ema_records.items():
        plt.plot(step_records, values, "--", label=f"EMA (sigma_rel={sigma_rel})")

    plt.xlabel("Step")
    plt.ylabel("Parameter Value")
    plt.title("Effect of Different sigma_rel Values on EMA")
    plt.legend()
    plt.grid(True)
    plt.savefig("ema_parameter_comparison.png")
    plt.close()
    """
