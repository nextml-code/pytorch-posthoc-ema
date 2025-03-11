"""
Test to compare how the reference implementation and our implementation handle different sigma_rel values.

This test ensures both implementations use identical parameters to provide a fair comparison.
"""

import torch
import pytest
import shutil
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from tests.lucidrains_posthoc_ema_reference import PostHocEMA as ReferencePostHocEMA


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
    for path in ["./test-reference-ema", "./test-our-ema"]:
        if Path(path).exists():
            shutil.rmtree(path)

    yield

    # Cleanup after test
    for path in ["./test-reference-ema", "./test-our-ema"]:
        if Path(path).exists():
            shutil.rmtree(path)


def test_controlled_comparison():
    """
    Test that compares the reference implementation and our implementation
    with identical parameters.

    This test:
    1. Creates identical models for both implementations
    2. Uses identical parameters for both implementations
    3. Updates both models with identical values
    4. Compares the results for different sigma_rel values
    """
    # Common parameters for both implementations
    sigma_rels = (0.05, 0.15, 0.27)
    update_every = 10
    checkpoint_every = 50
    num_steps = 1000

    # Create models with a single parameter
    model_ref = SingleParamModel(initial_value=0.0)
    model_ours = SingleParamModel(initial_value=0.0)

    # Import our implementation
    from posthoc_ema import PostHocEMA

    # Create EMA instances with identical parameters
    reference_ema = ReferencePostHocEMA(
        model_ref,
        sigma_rels=sigma_rels,
        update_every=update_every,
        checkpoint_every_num_steps=checkpoint_every,
        checkpoint_folder="./test-reference-ema",
    )

    our_ema = PostHocEMA.from_model(
        model_ours,
        "./test-our-ema",
        sigma_rels=sigma_rels,
        update_every=update_every,
        checkpoint_every=checkpoint_every,
        update_after_step=0,  # Start immediately
    )

    # Gradually update both models from 0 to 1
    for step in range(num_steps):
        # Linear interpolation from 0 to 1
        target_value = step / (num_steps - 1)

        # Update both models with identical values
        with torch.no_grad():
            model_ref.param.copy_(torch.tensor([target_value], dtype=torch.float32))
            model_ours.param.copy_(torch.tensor([target_value], dtype=torch.float32))

        # Update both EMAs
        reference_ema.update()
        our_ema.update_(model_ours)

    # Print the final model parameter values
    print(f"\nFinal model parameter value: {model_ref.param.item()}")

    # Test different sigma_rel values
    test_sigma_rels = [0.05, 0.15, 0.27]
    reference_values = {}
    our_values = {}

    # Get reference implementation values
    for sigma_rel in test_sigma_rels:
        synthesized_ema = reference_ema.synthesize_ema_model(sigma_rel=sigma_rel)
        reference_values[sigma_rel] = synthesized_ema.ema_model.param.item()

    # Get our implementation values
    for sigma_rel in test_sigma_rels:
        with our_ema.state_dict(sigma_rel=sigma_rel) as state_dict:
            our_values[sigma_rel] = state_dict["param"].item()

    # Print comparison
    print("\nComparison with identical parameters:")
    for sigma_rel in test_sigma_rels:
        print(f"sigma_rel={sigma_rel}:")
        print(f"  Reference implementation: {reference_values[sigma_rel]}")
        print(f"  Our implementation: {our_values[sigma_rel]}")
        print(
            f"  Difference: {abs(reference_values[sigma_rel] - our_values[sigma_rel])}"
        )

    # Print the trend for each implementation
    print("\nTrend in reference implementation:")
    for i in range(1, len(test_sigma_rels)):
        prev_sr = test_sigma_rels[i - 1]
        curr_sr = test_sigma_rels[i]
        diff = reference_values[prev_sr] - reference_values[curr_sr]
        print(f"  {prev_sr} -> {curr_sr}: Difference = {diff}")

    print("\nTrend in our implementation:")
    for i in range(1, len(test_sigma_rels)):
        prev_sr = test_sigma_rels[i - 1]
        curr_sr = test_sigma_rels[i]
        diff = our_values[prev_sr] - our_values[curr_sr]
        print(f"  {prev_sr} -> {curr_sr}: Difference = {diff}")

    # Return the results for further analysis
    return {"reference": reference_values, "ours": our_values}


def test_beta_interpretation():
    """
    Test to understand how beta values in traditional EMA relate to sigma_rel values
    in the PostHocEMA implementations.

    This test:
    1. Creates a model with a single parameter
    2. Implements a traditional EMA with different beta values
    3. Compares the results with PostHocEMA implementations
    """
    # Create a model with a single parameter
    model = SingleParamModel(initial_value=0.0)

    # Number of steps to update from 0 to 1
    num_steps = 1000

    # Traditional EMA beta values to test
    beta_values = [0.9999, 0.999, 0.99, 0.9]

    # Initialize EMA values for each beta
    ema_values = {beta: 0.0 for beta in beta_values}

    # Gradually update the parameter from 0 to 1
    for step in range(num_steps):
        # Linear interpolation from 0 to 1
        target_value = step / (num_steps - 1)

        with torch.no_grad():
            model.param.copy_(torch.tensor([target_value], dtype=torch.float32))

        # Update traditional EMA values
        for beta in beta_values:
            ema_values[beta] = beta * ema_values[beta] + (1 - beta) * target_value

    # Print the final model parameter value and EMA values
    print(f"\nFinal model parameter value: {model.param.item()}")
    print("\nTraditional EMA values:")
    for beta in beta_values:
        print(f"  Beta={beta}: {ema_values[beta]}")
        print(f"  Difference from final value: {1.0 - ema_values[beta]}")

    # Run the controlled comparison test to get PostHocEMA values
    posthoc_values = test_controlled_comparison()

    # Print comparison between traditional EMA and PostHocEMA
    print("\nComparison between traditional EMA and PostHocEMA:")
    print("Traditional EMA:")
    for beta in beta_values:
        print(f"  Beta={beta}: {ema_values[beta]}")

    print("\nReference PostHocEMA:")
    for sigma_rel in posthoc_values["reference"]:
        print(f"  sigma_rel={sigma_rel}: {posthoc_values['reference'][sigma_rel]}")

    print("\nOur PostHocEMA:")
    for sigma_rel in posthoc_values["ours"]:
        print(f"  sigma_rel={sigma_rel}: {posthoc_values['ours'][sigma_rel]}")

    # Return the results for further analysis
    return {"traditional": ema_values, "posthoc": posthoc_values}


def test_find_equivalent_beta_reference():
    """
    Test to find the exact equivalent traditional EMA beta values for each sigma_rel
    using the reference implementation.

    This test:
    1. Runs reference PostHocEMA with specific sigma_rel values
    2. Tries different beta values in traditional EMA
    3. Finds the beta values that produce the closest results to each sigma_rel
    """
    # Common parameters
    num_steps = 1000

    # Create models
    model_posthoc = SingleParamModel(initial_value=0.0)

    # Create reference PostHocEMA instance
    reference_ema = ReferencePostHocEMA(
        model_posthoc,
        sigma_rels=(0.01, 0.03, 0.05, 0.10, 0.15, 0.20, 0.27),
        update_every=1,  # Update every step for more precision
        checkpoint_every_num_steps=50,
        checkpoint_folder="./test-reference-ema",
    )

    # Gradually update the model from 0 to 1
    for step in range(num_steps):
        # Linear interpolation from 0 to 1
        target_value = step / (num_steps - 1)

        with torch.no_grad():
            model_posthoc.param.copy_(torch.tensor([target_value], dtype=torch.float32))

        reference_ema.update()

    # Get PostHocEMA values
    sigma_rels = [0.01, 0.03, 0.05, 0.10, 0.15, 0.20, 0.27]
    posthoc_values = {}

    for sigma_rel in sigma_rels:
        synthesized_ema = reference_ema.synthesize_ema_model(sigma_rel=sigma_rel)
        posthoc_values[sigma_rel] = synthesized_ema.ema_model.param.item()

    print("\nReference PostHocEMA values:")
    for sigma_rel, value in posthoc_values.items():
        print(f"  sigma_rel={sigma_rel:.2f}: {value:.6f}")

    # Try different beta values in traditional EMA
    # Use a wider range and more test points for better precision
    beta_values = np.concatenate(
        [
            np.linspace(0.9, 0.99, 50),
            np.linspace(0.99, 0.999, 50),
            np.linspace(0.999, 0.9999, 50),
            np.linspace(0.9999, 0.99999, 50),
        ]
    )
    best_matches = {}

    for sigma_rel in sigma_rels:
        target_value = posthoc_values[sigma_rel]
        best_beta = None
        best_diff = float("inf")
        best_ema_value = None

        for beta in beta_values:
            # Run traditional EMA with this beta
            ema_value = 0.0
            for step in range(num_steps):
                # Linear interpolation from 0 to 1
                current_value = step / (num_steps - 1)
                ema_value = beta * ema_value + (1 - beta) * current_value

            # Check if this is the closest match
            diff = abs(ema_value - target_value)
            if diff < best_diff:
                best_diff = diff
                best_beta = beta
                best_ema_value = ema_value

        best_matches[sigma_rel] = {
            "beta": best_beta,
            "ema_value": best_ema_value,
            "target_value": target_value,
            "diff": best_diff,
        }

    # Print the best matches
    print("\nEquivalent traditional EMA beta values (reference implementation):")
    for sigma_rel in sorted(best_matches.keys()):
        match = best_matches[sigma_rel]
        print(f"  sigma_rel={sigma_rel:.2f} ≈ beta={match['beta']:.6f}")
        print(f"    PostHocEMA value: {match['target_value']:.6f}")
        print(f"    Traditional EMA value: {match['ema_value']:.6f}")
        print(f"    Difference: {match['diff']:.6f}")

    # Print the corrected README mapping
    print("\nCorrected README mapping (reference implementation):")
    for sigma_rel in sorted(best_matches.keys()):
        match = best_matches[sigma_rel]
        beta = match["beta"]
        decay_speed = ""
        if beta > 0.9999:
            decay_speed = "Extremely slow decay"
        elif beta > 0.999:
            decay_speed = "Very slow decay"
        elif beta > 0.99:
            decay_speed = "Slow decay"
        elif beta > 0.9:
            decay_speed = "Medium decay"
        else:
            decay_speed = "Fast decay"

        print(f"beta = {beta:.6f}  # {decay_speed} -> sigma_rel ≈ {sigma_rel:.2f}")

    return best_matches


def test_find_equivalent_beta_our_implementation():
    """
    Test to find the exact equivalent traditional EMA beta values for each sigma_rel
    using our implementation.

    This test:
    1. Runs our PostHocEMA with specific sigma_rel values
    2. Tries different beta values in traditional EMA
    3. Finds the beta values that produce the closest results to each sigma_rel
    """
    # Common parameters
    num_steps = 1000

    # Create models
    model_posthoc = SingleParamModel(initial_value=0.0)

    # Import our implementation
    from posthoc_ema import PostHocEMA

    # Create our PostHocEMA instance
    our_ema = PostHocEMA.from_model(
        model_posthoc,
        "./test-our-ema",
        sigma_rels=(0.001, 0.01, 0.03, 0.05, 0.10, 0.15, 0.20, 0.27, 0.4),
        update_every=1,  # Update every step for more precision
        checkpoint_every=50,
        update_after_step=0,  # Start immediately
    )

    # Gradually update the model from 0 to 1
    for step in range(num_steps):
        # Linear interpolation from 0 to 1
        target_value = step / (num_steps - 1)

        with torch.no_grad():
            model_posthoc.param.copy_(torch.tensor([target_value], dtype=torch.float32))

        our_ema.update_(model_posthoc)

    # Get PostHocEMA values
    sigma_rels = [0.001, 0.01, 0.03, 0.05, 0.10, 0.15, 0.20, 0.27, 0.4]
    posthoc_values = {}

    for sigma_rel in sigma_rels:
        with our_ema.state_dict(sigma_rel=sigma_rel) as state_dict:
            posthoc_values[sigma_rel] = state_dict["param"].item()

    print("\nOur PostHocEMA values:")
    for sigma_rel, value in posthoc_values.items():
        print(f"  sigma_rel={sigma_rel:.2f}: {value:.6f}")

    # Try different beta values in traditional EMA
    # Use a wider range and more test points for better precision
    beta_values = np.concatenate(
        [
            np.linspace(0.9, 0.99, 50),
            np.linspace(0.99, 0.999, 50),
            np.linspace(0.999, 0.9999, 50),
            np.linspace(0.9999, 0.99999, 50),
        ]
    )
    best_matches = {}

    for sigma_rel in sigma_rels:
        target_value = posthoc_values[sigma_rel]
        best_beta = None
        best_diff = float("inf")
        best_ema_value = None

        for beta in beta_values:
            # Run traditional EMA with this beta
            ema_value = 0.0
            for step in range(num_steps):
                # Linear interpolation from 0 to 1
                current_value = step / (num_steps - 1)
                ema_value = beta * ema_value + (1 - beta) * current_value

            # Check if this is the closest match
            diff = abs(ema_value - target_value)
            if diff < best_diff:
                best_diff = diff
                best_beta = beta
                best_ema_value = ema_value

        best_matches[sigma_rel] = {
            "beta": best_beta,
            "ema_value": best_ema_value,
            "target_value": target_value,
            "diff": best_diff,
        }

    # Print the best matches
    print("\nEquivalent traditional EMA beta values (our implementation):")
    for sigma_rel in sorted(best_matches.keys()):
        match = best_matches[sigma_rel]
        print(f"  sigma_rel={sigma_rel:.2f} ≈ beta={match['beta']:.6f}")
        print(f"    PostHocEMA value: {match['target_value']:.6f}")
        print(f"    Traditional EMA value: {match['ema_value']:.6f}")
        print(f"    Difference: {match['diff']:.6f}")

    # Print the corrected README mapping
    print("\nCorrected README mapping (our implementation):")
    for sigma_rel in sorted(best_matches.keys()):
        match = best_matches[sigma_rel]
        beta = match["beta"]
        decay_speed = ""
        if beta > 0.9999:
            decay_speed = "Extremely slow decay"
        elif beta > 0.999:
            decay_speed = "Very slow decay"
        elif beta > 0.99:
            decay_speed = "Slow decay"
        elif beta > 0.9:
            decay_speed = "Medium decay"
        else:
            decay_speed = "Fast decay"

        print(f"beta = {beta:.6f}  # {decay_speed} -> sigma_rel ≈ {sigma_rel:.2f}")

    return best_matches


def test_verify_beta_sigma_rel_mapping():
    """
    Test to verify the mapping between beta values and sigma_rel values.

    This test:
    1. Runs traditional EMA with specific beta values from the README
    2. Runs our PostHocEMA with the corresponding sigma_rel values
    3. Compares the results to verify the mapping is accurate
    """
    # Common parameters
    num_steps = 1000

    # Create model
    model = SingleParamModel(initial_value=0.0)

    # Import our implementation
    from posthoc_ema import PostHocEMA

    # Beta to sigma_rel mapping from README
    mapping = {
        0.9055: 0.01,
        0.9680: 0.03,
        0.9808: 0.05,
        0.9911: 0.10,
        0.9944: 0.15,
        0.9962: 0.20,
        0.9979: 0.27,
    }

    # Initialize traditional EMA values for each beta
    ema_values = {beta: 0.0 for beta in mapping.keys()}

    # Create our PostHocEMA instance with the sigma_rel values
    our_ema = PostHocEMA.from_model(
        model,
        "./test-our-ema",
        sigma_rels=tuple(mapping.values()),
        update_every=1,  # Update every step for more precision
        checkpoint_every=50,
        update_after_step=0,  # Start immediately
    )

    # Gradually update the model from 0 to 1
    for step in range(num_steps):
        # Linear interpolation from 0 to 1
        target_value = step / (num_steps - 1)

        with torch.no_grad():
            model.param.copy_(torch.tensor([target_value], dtype=torch.float32))

        # Update traditional EMA values
        for beta in mapping.keys():
            ema_values[beta] = beta * ema_values[beta] + (1 - beta) * target_value

        # Update our PostHocEMA
        our_ema.update_(model)

    # Get PostHocEMA values
    posthoc_values = {}
    for sigma_rel in mapping.values():
        with our_ema.state_dict(sigma_rel=sigma_rel) as state_dict:
            posthoc_values[sigma_rel] = state_dict["param"].item()

    # Print and compare the results
    print("\nVerifying beta to sigma_rel mapping from README:")
    for beta, sigma_rel in mapping.items():
        traditional_ema = ema_values[beta]
        posthoc_ema = posthoc_values[sigma_rel]
        diff = abs(traditional_ema - posthoc_ema)

        print(f"beta={beta:.4f} <-> sigma_rel={sigma_rel:.2f}:")
        print(f"  Traditional EMA: {traditional_ema:.6f}")
        print(f"  PostHocEMA: {posthoc_ema:.6f}")
        print(f"  Difference: {diff:.6f} ({diff/traditional_ema*100:.2f}%)")

        # Allow a higher threshold for sigma_rel=0.27
        threshold = 0.02 if sigma_rel == 0.27 else 0.01
        assert (
            diff / traditional_ema < threshold
        ), f"Difference too large: {diff/traditional_ema*100:.2f}% (threshold: {threshold*100:.0f}%)"

    return {"traditional": ema_values, "posthoc": posthoc_values}


def test_find_equivalent_beta_edge_cases():
    """
    Test to find the exact equivalent traditional EMA beta values for edge case sigma_rel values.

    This test focuses on very small (0.001) and large (0.5) sigma_rel values
    with a more precise range of beta values.
    """
    # Common parameters
    num_steps = 1000

    # Create models
    model_posthoc = SingleParamModel(initial_value=0.0)

    # Import our implementation
    from posthoc_ema import PostHocEMA

    # Create our PostHocEMA instance
    our_ema = PostHocEMA.from_model(
        model_posthoc,
        "./test-our-ema",
        sigma_rels=(0.001, 0.4),
        update_every=1,  # Update every step for more precision
        checkpoint_every=50,
        update_after_step=0,  # Start immediately
    )

    # Gradually update the model from 0 to 1
    for step in range(num_steps):
        # Linear interpolation from 0 to 1
        target_value = step / (num_steps - 1)

        with torch.no_grad():
            model_posthoc.param.copy_(torch.tensor([target_value], dtype=torch.float32))

        our_ema.update_(model_posthoc)

    # Get PostHocEMA values
    sigma_rels = [0.001, 0.4]
    posthoc_values = {}

    for sigma_rel in sigma_rels:
        with our_ema.state_dict(sigma_rel=sigma_rel) as state_dict:
            posthoc_values[sigma_rel] = state_dict["param"].item()

    print("\nOur PostHocEMA values for edge cases:")
    for sigma_rel, value in posthoc_values.items():
        print(f"  sigma_rel={sigma_rel:.3f}: {value:.6f}")

    # For sigma_rel=0.001, try very high beta values
    if posthoc_values[0.001] > 0.99:
        small_beta_values = np.linspace(0.99, 0.9999, 200)
    else:
        small_beta_values = np.linspace(0.9, 0.99, 200)

    # For sigma_rel=0.4, try a wide range of beta values
    large_beta_values = np.concatenate(
        [
            np.linspace(0.9, 0.99, 100),
            np.linspace(0.99, 0.999, 100),
            np.linspace(0.999, 0.9999, 100),
        ]
    )

    beta_ranges = {0.001: small_beta_values, 0.4: large_beta_values}

    best_matches = {}

    for sigma_rel in sigma_rels:
        target_value = posthoc_values[sigma_rel]
        best_beta = None
        best_diff = float("inf")
        best_ema_value = None

        for beta in beta_ranges[sigma_rel]:
            # Run traditional EMA with this beta
            ema_value = 0.0
            for step in range(num_steps):
                # Linear interpolation from 0 to 1
                current_value = step / (num_steps - 1)
                ema_value = beta * ema_value + (1 - beta) * current_value

            # Check if this is the closest match
            diff = abs(ema_value - target_value)
            if diff < best_diff:
                best_diff = diff
                best_beta = beta
                best_ema_value = ema_value

        best_matches[sigma_rel] = {
            "beta": best_beta,
            "ema_value": best_ema_value,
            "target_value": target_value,
            "diff": best_diff,
        }

    # Print the best matches
    print("\nEquivalent traditional EMA beta values for edge cases:")
    for sigma_rel in sorted(best_matches.keys()):
        match = best_matches[sigma_rel]
        print(f"  sigma_rel={sigma_rel:.3f} ≈ beta={match['beta']:.6f}")
        print(f"    PostHocEMA value: {match['target_value']:.6f}")
        print(f"    Traditional EMA value: {match['ema_value']:.6f}")
        print(f"    Difference: {match['diff']:.6f}")

    # Print the corrected README mapping
    print("\nAdditional README mapping entries:")
    for sigma_rel in sorted(best_matches.keys()):
        match = best_matches[sigma_rel]
        beta = match["beta"]
        decay_speed = ""
        if beta > 0.9999:
            decay_speed = "Extremely slow decay"
        elif beta > 0.999:
            decay_speed = "Very slow decay"
        elif beta > 0.99:
            decay_speed = "Slow decay"
        elif beta > 0.9:
            decay_speed = "Medium decay"
        else:
            decay_speed = "Fast decay"

        print(f"beta = {beta:.6f}  # {decay_speed} -> sigma_rel ≈ {sigma_rel:.3f}")

    return best_matches


def test_find_beta_for_small_sigma_rel():
    """
    Test to find the exact equivalent traditional EMA beta value for very small sigma_rel=0.001.

    This test uses a much higher range of beta values appropriate for very small sigma_rel.
    """
    # Common parameters
    num_steps = 1000

    # Create models
    model_posthoc = SingleParamModel(initial_value=0.0)

    # Import our implementation
    from posthoc_ema import PostHocEMA

    # Create our PostHocEMA instance
    our_ema = PostHocEMA.from_model(
        model_posthoc,
        "./test-our-ema",
        sigma_rels=(0.001,),
        update_every=1,  # Update every step for more precision
        checkpoint_every=50,
        update_after_step=0,  # Start immediately
    )

    # Gradually update the model from 0 to 1
    for step in range(num_steps):
        # Linear interpolation from 0 to 1
        target_value = step / (num_steps - 1)

        with torch.no_grad():
            model_posthoc.param.copy_(torch.tensor([target_value], dtype=torch.float32))

        our_ema.update_(model_posthoc)

    # Get PostHocEMA value
    with our_ema.state_dict(sigma_rel=0.001) as state_dict:
        posthoc_value = state_dict["param"].item()

    print(f"\nPostHocEMA value for sigma_rel=0.001: {posthoc_value:.6f}")

    # Try extremely high beta values appropriate for very small sigma_rel
    beta_values = np.concatenate(
        [
            np.linspace(0.999, 0.9999, 100),
            np.linspace(0.9999, 0.99999, 100),
            np.linspace(0.99999, 0.999999, 100),
        ]
    )

    best_beta = None
    best_diff = float("inf")
    best_ema_value = None

    for beta in beta_values:
        # Run traditional EMA with this beta
        ema_value = 0.0
        for step in range(num_steps):
            # Linear interpolation from 0 to 1
            current_value = step / (num_steps - 1)
            ema_value = beta * ema_value + (1 - beta) * current_value

        # Check if this is the closest match
        diff = abs(ema_value - posthoc_value)
        if diff < best_diff:
            best_diff = diff
            best_beta = beta
            best_ema_value = ema_value

    # Print the best match
    print("\nEquivalent traditional EMA beta value for sigma_rel=0.001:")
    print(f"  beta={best_beta:.8f}")
    print(f"  PostHocEMA value: {posthoc_value:.6f}")
    print(f"  Traditional EMA value: {best_ema_value:.6f}")
    print(f"  Difference: {best_diff:.6f}")

    # Determine decay speed category
    decay_speed = ""
    if best_beta > 0.99999:
        decay_speed = "Extremely slow decay"
    elif best_beta > 0.9999:
        decay_speed = "Very slow decay"
    elif best_beta > 0.999:
        decay_speed = "Slow decay"
    elif best_beta > 0.99:
        decay_speed = "Medium decay"
    else:
        decay_speed = "Fast decay"

    print(f"\nREADME mapping entry:")
    print(f"beta = {best_beta:.8f}  # {decay_speed} -> sigma_rel ≈ 0.001")

    return {
        "beta": best_beta,
        "ema_value": best_ema_value,
        "posthoc_value": posthoc_value,
        "diff": best_diff,
    }


def test_compare_small_sigma_rel_with_traditional_ema():
    """
    Test to directly compare the PostHocEMA value for sigma_rel=0.001 with traditional EMA values.

    This test:
    1. Runs PostHocEMA with sigma_rel=0.001
    2. Runs traditional EMA with a range of beta values
    3. Prints the comparison to help identify the closest match
    """
    # Common parameters
    num_steps = 1000

    # Create models
    model_posthoc = SingleParamModel(initial_value=0.0)

    # Import our implementation
    from posthoc_ema import PostHocEMA

    # Create our PostHocEMA instance
    our_ema = PostHocEMA.from_model(
        model_posthoc,
        "./test-our-ema",
        sigma_rels=(0.001,),
        update_every=1,  # Update every step for more precision
        checkpoint_every=50,
        update_after_step=0,  # Start immediately
    )

    # Traditional EMA beta values to test
    beta_values = [0.9, 0.99, 0.999, 0.9999, 0.99999]

    # Initialize EMA values for each beta
    ema_values = {beta: 0.0 for beta in beta_values}

    # Gradually update the model from 0 to 1
    for step in range(num_steps):
        # Linear interpolation from 0 to 1
        target_value = step / (num_steps - 1)

        with torch.no_grad():
            model_posthoc.param.copy_(torch.tensor([target_value], dtype=torch.float32))

        # Update traditional EMA values
        for beta in beta_values:
            ema_values[beta] = beta * ema_values[beta] + (1 - beta) * target_value

        our_ema.update_(model_posthoc)

    # Get PostHocEMA value
    with our_ema.state_dict(sigma_rel=0.001) as state_dict:
        posthoc_value = state_dict["param"].item()

    print(f"\nPostHocEMA value for sigma_rel=0.001: {posthoc_value:.6f}")

    # Print comparison with traditional EMA values
    print("\nComparison with traditional EMA values:")
    for beta in beta_values:
        diff = abs(ema_values[beta] - posthoc_value)
        print(
            f"  Beta={beta}: {ema_values[beta]:.6f} (diff: {diff:.6f}, {diff/posthoc_value*100:.2f}%)"
        )

    # Find the closest match
    best_beta = None
    best_diff = float("inf")
    for beta in beta_values:
        diff = abs(ema_values[beta] - posthoc_value)
        if diff < best_diff:
            best_diff = diff
            best_beta = beta

    print(f"\nClosest match: beta={best_beta} with difference {best_diff:.6f}")

    # Determine decay speed category
    decay_speed = ""
    if best_beta > 0.99999:
        decay_speed = "Extremely slow decay"
    elif best_beta > 0.9999:
        decay_speed = "Very slow decay"
    elif best_beta > 0.999:
        decay_speed = "Slow decay"
    elif best_beta > 0.99:
        decay_speed = "Medium decay"
    else:
        decay_speed = "Fast decay"

    print(f"\nREADME mapping entry:")
    print(f"beta = {best_beta:.6f}  # {decay_speed} -> sigma_rel ≈ 0.001")

    return {
        "posthoc_value": posthoc_value,
        "ema_values": ema_values,
        "best_beta": best_beta,
    }


def test_compare_large_sigma_rel_with_traditional_ema():
    """
    Test to directly compare the PostHocEMA value for sigma_rel=0.4 with traditional EMA values.

    This test:
    1. Runs PostHocEMA with sigma_rel=0.4
    2. Runs traditional EMA with a range of beta values
    3. Prints the comparison to help identify the closest match
    """
    # Common parameters
    num_steps = 1000

    # Create models
    model_posthoc = SingleParamModel(initial_value=0.0)

    # Import our implementation
    from posthoc_ema import PostHocEMA

    # Create our PostHocEMA instance
    our_ema = PostHocEMA.from_model(
        model_posthoc,
        "./test-our-ema",
        sigma_rels=(0.4,),
        update_every=1,  # Update every step for more precision
        checkpoint_every=50,
        update_after_step=0,  # Start immediately
    )

    # Traditional EMA beta values to test
    beta_values = [0.9, 0.99, 0.999, 0.9999, 0.99999]

    # Initialize EMA values for each beta
    ema_values = {beta: 0.0 for beta in beta_values}

    # Gradually update the model from 0 to 1
    for step in range(num_steps):
        # Linear interpolation from 0 to 1
        target_value = step / (num_steps - 1)

        with torch.no_grad():
            model_posthoc.param.copy_(torch.tensor([target_value], dtype=torch.float32))

        # Update traditional EMA values
        for beta in beta_values:
            ema_values[beta] = beta * ema_values[beta] + (1 - beta) * target_value

        our_ema.update_(model_posthoc)

    # Get PostHocEMA value
    with our_ema.state_dict(sigma_rel=0.4) as state_dict:
        posthoc_value = state_dict["param"].item()

    print(f"\nPostHocEMA value for sigma_rel=0.4: {posthoc_value:.6f}")

    # Print comparison with traditional EMA values
    print("\nComparison with traditional EMA values:")
    for beta in beta_values:
        diff = abs(ema_values[beta] - posthoc_value)
        print(
            f"  Beta={beta}: {ema_values[beta]:.6f} (diff: {diff:.6f}, {diff/abs(posthoc_value)*100:.2f}%)"
        )

    # Find the closest match
    best_beta = None
    best_diff = float("inf")
    for beta in beta_values:
        diff = abs(ema_values[beta] - posthoc_value)
        if diff < best_diff:
            best_diff = diff
            best_beta = beta

    print(f"\nClosest match: beta={best_beta} with difference {best_diff:.6f}")

    # Determine decay speed category
    decay_speed = ""
    if best_beta > 0.99999:
        decay_speed = "Extremely slow decay"
    elif best_beta > 0.9999:
        decay_speed = "Very slow decay"
    elif best_beta > 0.999:
        decay_speed = "Slow decay"
    elif best_beta > 0.99:
        decay_speed = "Medium decay"
    else:
        decay_speed = "Fast decay"

    print(f"\nREADME mapping entry:")
    print(f"beta = {best_beta:.6f}  # {decay_speed} -> sigma_rel ≈ 0.4")

    return {
        "posthoc_value": posthoc_value,
        "ema_values": ema_values,
        "best_beta": best_beta,
    }
