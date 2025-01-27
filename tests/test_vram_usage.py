from pathlib import Path

import psutil
import torch
import torch.cuda
import torch.nn as nn

from posthoc_ema import PostHocEMA

"""
PostHocEMA Configuration Options:
--------------------------------
- checkpoint_dir: Directory to store checkpoints
- max_checkpoints: Maximum number of checkpoints to keep per EMA model (default=100)
- sigma_rels: Relative standard deviations for EMA models (default=(0.05, 0.28))
- update_every: Number of steps between EMA updates (default=10)
- checkpoint_every: Number of steps between checkpoints (default=1000)
- checkpoint_dtype: Data type for checkpoint storage (default=torch.float16)

Example usage with configuration:
--------------------------------
posthoc_ema = PostHocEMA.from_model(
    model,
    checkpoint_dir="path/to/checkpoints",
    max_checkpoints=50,  # Keep last 50 checkpoints per EMA model
    sigma_rels=(0.15, 0.25),  # Custom relative standard deviations
    update_every=5,  # Update EMA weights every 5 steps
    checkpoint_every=500,  # Create checkpoints every 500 steps
    checkpoint_dtype=torch.float32,  # Store checkpoints in full precision
)
"""


def get_gpu_memory_usage():
    """Get current GPU memory usage in MB."""
    if not torch.cuda.is_available():
        return 0
    return torch.cuda.memory_allocated() / 1024 / 1024


def get_gpu_max_memory_usage():
    """Get maximum GPU memory usage since last reset in MB."""
    if not torch.cuda.is_available():
        return 0
    return torch.cuda.max_memory_allocated() / 1024 / 1024


def reset_gpu_max_memory():
    """Reset peak memory tracking."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def get_ram_usage():
    """Get current RAM usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def get_module_device(model: nn.Module) -> str:
    """Get the device of a model by checking its first parameter."""
    return next(model.parameters()).device.type


def test_vram_usage_with_classifier():
    """Test VRAM usage before and after PostHocEMA initialization."""
    if not torch.cuda.is_available():
        return

    # Clear cache and get initial memory state
    torch.cuda.empty_cache()
    reset_gpu_max_memory()
    initial_memory = get_gpu_memory_usage()
    initial_ram = get_ram_usage()
    print(f"\nInitial state:")
    print(f"VRAM: {initial_memory:.2f}MB")
    print(f"RAM:  {initial_ram:.2f}MB")

    # Load a much larger model to GPU to make VRAM issues more apparent
    model = nn.Sequential(
        nn.Linear(2048, 2048),
        nn.ReLU(),
        nn.Linear(2048, 2048),
        nn.ReLU(),
        nn.Linear(2048, 2048),
        nn.ReLU(),
        nn.Linear(2048, 2048),
    ).cuda()
    model_memory = get_gpu_memory_usage()
    model_ram = get_ram_usage()
    print(f"\nAfter model creation:")
    print(f"VRAM: {model_memory:.2f}MB")
    print(f"RAM:  {model_ram:.2f}MB")
    print(f"Model size in VRAM: {model_memory - initial_memory:.2f}MB")

    # Monitor VRAM during PostHocEMA initialization
    print("\nStarting PostHocEMA initialization...")
    pre_init_memory = get_gpu_memory_usage()
    reset_gpu_max_memory()  # Reset peak memory tracking before initialization
    save_path = Path("test_ema_checkpoint")

    # This should cause a VRAM spike due to model copying before CPU transfer
    posthoc_ema = PostHocEMA.from_model(
        model,
        save_path,
        max_checkpoints=2,
        sigma_rels=(0.05, 0.15),
        update_every=1,
        checkpoint_every=5,
        checkpoint_dtype=torch.float32,
    )

    # Check memory right after initialization
    post_init_memory = get_gpu_memory_usage()
    post_init_ram = get_ram_usage()
    peak_memory = get_gpu_max_memory_usage()
    print(f"\nAfter EMA initialization:")
    print(f"VRAM: {post_init_memory:.2f}MB")
    print(f"Peak VRAM during init: {peak_memory:.2f}MB")
    print(f"RAM:  {post_init_ram:.2f}MB")
    print(f"VRAM spike during init: {peak_memory - pre_init_memory:.2f}MB")

    # This assertion should fail because of the VRAM spike during initialization
    assert peak_memory <= pre_init_memory + 1.0, (
        f"EMA initialization caused significant VRAM spike. "
        f"Pre-init: {pre_init_memory:.2f}MB, Peak: {peak_memory:.2f}MB"
    )

    # Verify EMA models are on CPU
    for ema_model in posthoc_ema.ema_models:
        for param in ema_model.ema_model.parameters():
            assert (
                param.device.type == "cpu"
            ), f"EMA model parameter found on {param.device.type}, should be on cpu"

    # Create some checkpoints
    for _ in range(5):
        with torch.no_grad():
            # Simulate training updates
            for param in model.parameters():
                param.add_(torch.randn_like(param) * 0.1)
            posthoc_ema.update_(model)

    # Check memory after creating checkpoints
    checkpoint_memory = get_gpu_memory_usage()
    checkpoint_ram = get_ram_usage()
    print(f"\nAfter creating checkpoints:")
    print(f"VRAM: {checkpoint_memory:.2f}MB")
    print(f"RAM:  {checkpoint_ram:.2f}MB")

    # Test state dict access doesn't increase VRAM
    state_dict = posthoc_ema.state_dict(sigma_rel=0.15)
    state_dict_memory = get_gpu_memory_usage()
    state_dict_ram = get_ram_usage()
    print(f"\nAfter state dict access:")
    print(f"VRAM: {state_dict_memory:.2f}MB")
    print(f"RAM:  {state_dict_ram:.2f}MB")

    assert state_dict_memory <= model_memory + 1.0, (
        f"Getting state dict changed VRAM usage. "
        f"Before: {model_memory:.2f}MB, After: {state_dict_memory:.2f}MB"
    )

    # Test context manager automatically moves model to CPU and handles VRAM correctly
    print(f"\nEntering context manager...")
    with posthoc_ema.model(model, sigma_rel=0.15) as ema_model:
        # Initial entry should not increase VRAM as PostHocEMA should move model to CPU
        entry_memory = get_gpu_memory_usage()
        entry_ram = get_ram_usage()
        print(f"\nAfter context entry (before cuda):")
        print(f"VRAM: {entry_memory:.2f}MB")
        print(f"RAM:  {entry_ram:.2f}MB")

        assert (
            entry_memory == 0
        ), f"PostHocEMA failed to move model to CPU: {entry_memory:.2f}MB VRAM in use"
        assert (
            get_module_device(model) == "cpu"
        ), "Model should be automatically moved to CPU"

        # Only after explicitly moving to GPU should we see VRAM usage
        ema_model.cuda()
        context_memory = get_gpu_memory_usage()
        context_ram = get_ram_usage()
        print(f"\nAfter moving EMA model to CUDA:")
        print(f"VRAM: {context_memory:.2f}MB")
        print(f"RAM:  {context_ram:.2f}MB")

        assert context_memory > 0, "EMA model should use VRAM after moving to GPU"

    # After context exit, memory should return to original model memory
    # since the model is moved back to its original device (GPU)
    post_context_memory = get_gpu_memory_usage()
    post_context_ram = get_ram_usage()
    print(f"\nAfter context exit:")
    print(f"VRAM: {post_context_memory:.2f}MB")
    print(f"RAM:  {post_context_ram:.2f}MB")

    assert post_context_memory == model_memory, (
        f"Context manager didn't restore original VRAM usage. "
        f"Expected: {model_memory:.2f}MB, Got: {post_context_memory:.2f}MB"
    )

    # Cleanup
    model.cpu()  # Move model to CPU before deletion
    del model
    del posthoc_ema
    if save_path.exists():
        for file in save_path.glob("*"):
            file.unlink()
        save_path.rmdir()

    # Force CUDA cleanup
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Final memory state
    final_memory = get_gpu_memory_usage()
    final_ram = get_ram_usage()
    print(f"\nFinal state after cleanup:")
    print(f"VRAM: {final_memory:.2f}MB")
    print(f"RAM:  {final_ram:.2f}MB")

    # Verify cleanup was successful
    assert (
        final_memory == 0
    ), f"Failed to cleanup CUDA memory. Still using {final_memory:.2f}MB VRAM"
