"""Test to verify the relationship between beta and sigma_rel values."""

import pytest
from posthoc_ema.utils import beta_to_sigma_rel, sigma_rel_to_beta


def test_beta_to_sigma_rel_relationship():
    """Test the relationship between beta and sigma_rel values as described in the README."""

    # Test cases from README
    test_cases = [
        (0.9999, 0.01),  # Very slow decay
        (0.9990, 0.03),  # Slow decay
        (0.9900, 0.10),  # Medium decay
        (0.9000, 0.27),  # Fast decay
    ]

    for beta, expected_sigma_rel in test_cases:
        # Convert beta to sigma_rel
        calculated_sigma_rel = beta_to_sigma_rel(beta)
        print(
            f"Beta: {beta:.4f} -> Calculated sigma_rel: {calculated_sigma_rel:.4f} (Expected: {expected_sigma_rel:.4f})"
        )

        # Check if the calculated sigma_rel is close to the expected value
        assert (
            abs(calculated_sigma_rel - expected_sigma_rel) < 0.02
        ), f"Beta {beta} should give sigma_rel close to {expected_sigma_rel}, got {calculated_sigma_rel}"


def test_sigma_rel_to_beta_relationship():
    """Test the conversion from sigma_rel to beta."""

    # Test cases
    sigma_rels = [0.01, 0.03, 0.10, 0.27]

    for sigma_rel in sigma_rels:
        beta = sigma_rel_to_beta(sigma_rel)
        print(f"sigma_rel: {sigma_rel:.4f} -> beta: {beta:.6f}")

        # Calculate effective half-life (number of steps to decay by half)
        # For EMA with decay rate beta, the half-life is approximately log(0.5)/log(beta)
        if beta > 0:
            half_life = -1 * (0.693147 / (1 - beta))  # log(0.5) ≈ -0.693147
            print(f"  Half-life: {half_life:.1f} steps")

        # Calculate effective window size (number of steps that contribute significantly)
        # For EMA with decay rate beta, the effective window size is approximately 1/(1-beta)
        window_size = 1 / (1 - beta)
        print(f"  Effective window size: {window_size:.1f} steps")


def test_sigma_rel_ordering():
    """Test that smaller sigma_rel values correspond to higher beta values (slower decay)."""

    sigma_rels = [0.01, 0.03, 0.10, 0.27]
    betas = [sigma_rel_to_beta(sr) for sr in sigma_rels]

    print("\nRelationship between sigma_rel and beta:")
    for sr, beta in zip(sigma_rels, betas):
        print(f"sigma_rel: {sr:.2f} -> beta: {beta:.6f}")

    # Check that beta values decrease as sigma_rel increases
    for i in range(1, len(betas)):
        assert (
            betas[i] < betas[i - 1]
        ), f"Beta for sigma_rel={sigma_rels[i]} should be less than beta for sigma_rel={sigma_rels[i-1]}"

    print(
        "\nThis confirms that smaller sigma_rel values correspond to higher beta values (slower decay)."
    )
    print(
        "In other words, as sigma_rel increases, the EMA adapts more quickly to recent values."
    )
