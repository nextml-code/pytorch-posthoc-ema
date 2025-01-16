from pathlib import Path

import torch
from torch import nn

from posthoc_ema import PostHocEMA


def test_training_scenario(tmp_path: Path):
    """Test PostHocEMA in a simulated training scenario.
    
    This test simulates:
    1. Regular model updates during training
    2. EMA updates
    3. Synthesizing EMA model during training for evaluation
    4. Resetting model weights to EMA state each epoch
    """
    # Setup
    model = nn.Linear(512, 512)
    posthoc_ema = PostHocEMA.from_model(
        model,
        checkpoint_dir=tmp_path / "posthoc-ema",
        max_checkpoints=5,  # Small number for test
        update_every=2,  # Update frequently for test
        checkpoint_every=10,  # Checkpoint frequently for test
    )
    
    # Training data
    data = torch.randn(1, 512)
    initial_prediction = model(data).detach().clone()
    
    # Simulate 2 epochs of training
    for epoch in range(2):
        # Training steps
        for step in range(20):  # Small number of steps for test
            # Simulate parameter updates (normally done by optimizer)
            with torch.no_grad():
                model.weight.copy_(torch.randn_like(model.weight))
                model.bias.copy_(torch.randn_like(model.bias))
            
            # Update EMA
            posthoc_ema.update_(model)
            
            # Every 5 steps, test synthesizing EMA model for evaluation
            # But only after we have at least one checkpoint
            if step % 5 == 0 and step >= posthoc_ema.checkpoint_every:
                with posthoc_ema.model(model, sigma_rel=0.15) as ema_model:
                    ema_prediction = ema_model(data)
                    # Verify EMA prediction is different from initial
                    assert not torch.allclose(ema_prediction, initial_prediction)
        
        # At end of epoch, reset model weights to EMA state
        # But only if we have checkpoints
        if (epoch + 1) * 20 >= posthoc_ema.checkpoint_every:
            with posthoc_ema.state_dict(sigma_rel=0.15) as ema_state_dict:
                model.load_state_dict(ema_state_dict, strict=False)
                
                # Verify model now gives same prediction as EMA
                model_prediction = model(data)
                with posthoc_ema.model(model, sigma_rel=0.15) as ema_model:
                    ema_prediction = ema_model(data)
                    assert torch.allclose(model_prediction, ema_prediction) 