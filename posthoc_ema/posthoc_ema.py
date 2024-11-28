from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Iterator

import torch
from torch import nn

from .karras_ema import KarrasEMA
from .utils import sigma_rel_to_gamma, solve_weights


class PostHocEMA:
    """
    Post-hoc EMA implementation with simplified interface and memory management.

    Args:
        model: The model to create EMAs of
        checkpoint_dir: Directory to store checkpoints
        max_checkpoints: Maximum number of checkpoints to keep per EMA model
        sigma_rels: Tuple of relative standard deviations for the maintained EMA models
        update_every: Number of steps between EMA updates
        checkpoint_every: Number of steps between checkpoints
        checkpoint_dtype: Data type for checkpoint storage
    """

    def __init__(
        self,
        model: nn.Module,
        checkpoint_dir: str | Path,
        max_checkpoints: int = 100,
        sigma_rels: tuple[float, ...] = (0.05, 0.28),
        update_every: int = 10,
        checkpoint_every: int = 1000,
        checkpoint_dtype: torch.dtype = torch.float16,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)

        self.max_checkpoints = max_checkpoints
        self.checkpoint_dtype = checkpoint_dtype
        self.update_every = update_every
        self.checkpoint_every = checkpoint_every

        # Initialize EMA models
        self.sigma_rels = sigma_rels
        self.gammas = tuple(map(sigma_rel_to_gamma, sigma_rels))
        
        # Create EMA models with shared online model reference
        self.ema_models = nn.ModuleList([
            KarrasEMA(
                model,
                sigma_rel=sigma_rel,
                update_every=update_every,
            ) for sigma_rel in sigma_rels
        ])

        self.step = 0

    def update(self, model: nn.Module) -> None:
        """
        Update EMA models and create checkpoints if needed.
        
        Args:
            model: Current state of the model to update EMAs with
        """
        # Update EMA models with current model state
        for ema_model in self.ema_models:
            # Update online model reference and copy parameters
            ema_model.online_model[0] = model
            if not ema_model.initted.item():
                ema_model.copy_params_from_model_to_ema()
                ema_model.initted.data.copy_(torch.tensor(True))
            ema_model.update()

        self.step += 1

        # Create checkpoint if needed
        if self.step % self.checkpoint_every == 0:
            self._create_checkpoint()
            self._cleanup_old_checkpoints()

    def _create_checkpoint(self) -> None:
        """Create checkpoints for all EMA models."""
        for idx, ema_model in enumerate(self.ema_models):
            filename = f"{idx}.{self.step}.pt"
            path = self.checkpoint_dir / filename

            state_dict = {
                k: v.to(self.checkpoint_dtype)
                for k, v in ema_model.state_dict().items()
            }
            torch.save(state_dict, path)

    def _cleanup_old_checkpoints(self) -> None:
        """Remove oldest checkpoints when exceeding max_checkpoints."""
        for idx in range(len(self.ema_models)):
            checkpoints = sorted(
                self.checkpoint_dir.glob(f"{idx}.*.pt"),
                key=lambda p: int(p.stem.split(".")[1]),
            )

            # Remove oldest checkpoints if exceeding limit
            while len(checkpoints) > self.max_checkpoints:
                checkpoints[0].unlink()
                checkpoints = checkpoints[1:]

    @contextmanager
    def model(
        self,
        base_model: nn.Module,
        sigma_rel: float,
        step: int | None = None,
    ) -> Iterator[nn.Module]:
        """
        Context manager for using synthesized EMA model.

        Args:
            base_model: Model to apply EMA weights to
            sigma_rel: Target relative standard deviation
            step: Optional specific training step to synthesize for

        Yields:
            nn.Module: Model with synthesized EMA weights
        """
        state_dict = self.state_dict(sigma_rel, step)
        ema_model = deepcopy(base_model)
        ema_model.load_state_dict(state_dict)

        try:
            yield ema_model
        finally:
            del ema_model

    def state_dict(
        self,
        sigma_rel: float,
        step: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Get state dict for synthesized EMA model.

        Args:
            sigma_rel: Target relative standard deviation
            step: Optional specific training step to synthesize for

        Returns:
            dict[str, torch.Tensor]: State dict with synthesized weights
        """
        # Convert target sigma_rel to gamma
        gamma = sigma_rel_to_gamma(sigma_rel)
        device = torch.device("cpu")  # Keep synthesis on CPU for memory efficiency

        # Get all checkpoints
        gammas = []
        timesteps = []
        checkpoints = []

        # Collect checkpoint info
        for idx in range(len(self.ema_models)):
            checkpoint_files = sorted(
                self.checkpoint_dir.glob(f"{idx}.*.pt"),
                key=lambda p: int(p.stem.split(".")[1]),
            )
            for file in checkpoint_files:
                _, timestep = map(int, file.stem.split("."))
                gammas.append(self.gammas[idx])
                timesteps.append(timestep)
                checkpoints.append(file)

        # Use latest step if not specified
        step = step if step is not None else max(timesteps)
        assert step <= max(
            timesteps
        ), f"Cannot synthesize for step {step} > max available step {max(timesteps)}"

        # Solve for optimal weights
        gamma_i = torch.tensor(gammas, device=device)
        t_i = torch.tensor(timesteps, device=device)
        gamma_r = torch.tensor([gamma], device=device)
        t_r = torch.tensor([step], device=device)

        weights = solve_weights(t_i, gamma_i, t_r, gamma_r)
        weights = weights.squeeze(-1)

        # Load first checkpoint to get state dict structure
        ckpt = torch.load(str(checkpoints[0]), map_location=device)
        
        # Extract just the model parameters (remove EMA-specific keys)
        model_keys = {k.replace("ema_model.", ""): k for k in ckpt.keys() 
                     if k.startswith("ema_model.")}
        
        # Zero initialize synthesized state
        synth_state = {
            k: torch.zeros_like(ckpt[v], device=device) 
            for k, v in model_keys.items()
        }

        # Combine checkpoints using solved weights
        for checkpoint, weight in zip(checkpoints, weights.tolist()):
            ckpt_state = torch.load(str(checkpoint), map_location=device)
            for k, v in model_keys.items():
                synth_state[k].add_(ckpt_state[v] * weight)

        return synth_state
