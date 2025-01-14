from pathlib import Path

import psutil
import torch
import torch.cuda
import torch.nn as nn

from posthoc_ema import PostHocEMA


def get_gpu_memory_usage():
    """Get current GPU memory usage in MB."""
    if not torch.cuda.is_available():
        return 0
    return torch.cuda.memory_allocated() / 1024 / 1024


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
    initial_memory = get_gpu_memory_usage()
    initial_ram = get_ram_usage()
    print(f"\nInitial state:")
    print(f"VRAM: {initial_memory:.2f}MB")
    print(f"RAM:  {initial_ram:.2f}MB")

    # Load model to GPU
    model = nn.Sequential(
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 10),
    ).cuda()
    model_memory = get_gpu_memory_usage()
    model_ram = get_ram_usage()
    print(f"\nAfter model creation:")
    print(f"VRAM: {model_memory:.2f}MB")
    print(f"RAM:  {model_ram:.2f}MB")

    # Initialize PostHocEMA
    save_path = Path("test_ema_checkpoint")
    posthoc_ema = PostHocEMA.from_model(
        model, 
        save_path,
        checkpoint_every=10,  # More frequent checkpoints for testing
    )
    
    # Check memory after initialization
    ema_memory = get_gpu_memory_usage()
    ema_ram = get_ram_usage()
    print(f"\nAfter EMA initialization:")
    print(f"VRAM: {ema_memory:.2f}MB")
    print(f"RAM:  {ema_ram:.2f}MB")

    # Memory after initialization should be exactly the same as model memory
    # as EMA weights are stored on CPU
    assert ema_memory == model_memory, (
        f"EMA initialization changed VRAM usage. "
        f"Before: {model_memory:.2f}MB, After: {ema_memory:.2f}MB"
    )

    # Verify EMA models are on CPU
    for ema_model in posthoc_ema.ema_models:
        for param in ema_model.ema_model.parameters():
            assert param.device.type == "cpu", (
                f"EMA model parameter found on {param.device.type}, should be on cpu"
            )

    # Create some checkpoints
    for _ in range(20):
        with torch.no_grad():
            # Simulate training updates
            for param in model.parameters():
                param.add_(torch.randn_like(param) * 0.1)
        posthoc_ema.update(model)

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

    assert state_dict_memory == model_memory, (
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

        assert entry_memory == 0, (
            f"PostHocEMA failed to move model to CPU: {entry_memory:.2f}MB VRAM in use"
        )
        assert get_module_device(model) == "cpu", "Model should be automatically moved to CPU"
        
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
    del model
    del posthoc_ema
    if save_path.exists():
        for file in save_path.glob("*"):
            file.unlink()
        save_path.rmdir()
    torch.cuda.empty_cache()

    # Final memory state
    final_memory = get_gpu_memory_usage()
    final_ram = get_ram_usage()
    print(f"\nFinal state after cleanup:")
    print(f"VRAM: {final_memory:.2f}MB")
    print(f"RAM:  {final_ram:.2f}MB")
