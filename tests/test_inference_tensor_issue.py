"""Tests for reproducing inference tensor error."""

import torch
import pytest
import shutil
from posthoc_ema import PostHocEMA
from pathlib import Path
from contextlib import contextmanager


@pytest.fixture(autouse=True)
def cleanup_checkpoints():
    """Clean up test checkpoints before and after each test."""
    # Cleanup before test
    for path in ["./test-checkpoints-inference"]:
        if Path(path).exists():
            shutil.rmtree(path)

    yield

    # Cleanup after test
    for path in ["./test-checkpoints-inference"]:
        if Path(path).exists():
            shutil.rmtree(path)


@contextmanager
def module_eval(model):
    """Temporarily set module in eval mode."""
    training = model.training
    model.eval()
    try:
        yield
    finally:
        model.train(training)


class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base_layer = torch.nn.Linear(512, 512)
        self.bn = torch.nn.BatchNorm1d(512)  # Add batch norm
        self.relu = torch.nn.ReLU()
        self.out_layer = torch.nn.Linear(512, 10)

    def forward(self, x):
        x = self.base_layer(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.out_layer(x)
        return x


def test_identical_predictions():
    """Test to reproduce identical predictions between EMA and non-EMA models."""
    # Create model
    model = SimpleModel()
    model.train()  # Start in training mode

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters())

    # Create EMA instance
    posthoc_ema = PostHocEMA.from_model(
        model,
        "test-checkpoints-inference",
        checkpoint_every=5,
        sigma_rels=(0.05, 0.28),
    )

    # First do some training to build up EMA weights
    for _ in range(10):
        x = torch.randn(32, 512, requires_grad=True)
        y = torch.randint(0, 10, (32,))

        optimizer.zero_grad()
        output = model(x)
        loss = torch.nn.functional.cross_entropy(output, y)
        loss.backward()
        optimizer.step()

        posthoc_ema.update_(model)

    # Create multiple test examples
    test_examples = [torch.randn(1, 512) for _ in range(16)]
    test_batch = torch.cat(test_examples)

    # Get predictions with EMA in eval mode
    with module_eval(model):
        with posthoc_ema.model(
            model, sigma_rel=0.4
        ) as ema_model:  # Use 0.4 like in the example
            ema_predictions = ema_model(test_batch)

    # Get predictions with raw model in eval mode
    with module_eval(model):
        raw_predictions = model(test_batch)

    # Check if predictions are identical
    print(
        f"Max difference between predictions: {(ema_predictions - raw_predictions).abs().max().item()}"
    )
    print(
        f"Mean difference between predictions: {(ema_predictions - raw_predictions).abs().mean().item()}"
    )
    print(
        f"Are predictions equal? {torch.allclose(ema_predictions, raw_predictions, rtol=1e-5, atol=1e-5)}"
    )

    # Also check individual examples
    for i in range(len(test_examples)):
        example_diff = (ema_predictions[i] - raw_predictions[i]).abs().max().item()
        print(f"Example {i} max difference: {example_diff}")

    assert not torch.allclose(
        ema_predictions, raw_predictions, rtol=1e-5, atol=1e-5
    ), "EMA and raw model predictions should not be identical"

    # Clean up
    if Path("test-checkpoints-inference").exists():
        for file in Path("test-checkpoints-inference").glob("*"):
            file.unlink()
        Path("test-checkpoints-inference").rmdir()
