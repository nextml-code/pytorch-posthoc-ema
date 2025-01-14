from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Iterator, Optional

import torch
from torch import nn

from .karras_ema import KarrasEMA
from .utils import sigma_rel_to_gamma, solve_weights


def _safe_torch_load(path: str | Path, *, map_location=None):
    """Helper function to load checkpoints with weights_only if supported."""
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        # Older PyTorch versions don't support weights_only
        return torch.load(path, map_location=map_location)


class PostHocEMA:
    """
    Post-hoc EMA implementation with simplified interface and memory management.
    
    Args:
        checkpoint_dir: Directory to store checkpoints
        max_checkpoints: Maximum number of checkpoints to keep per EMA model
        sigma_rels: Tuple of relative standard deviations for the maintained EMA models
        update_every: Number of steps between EMA updates
        checkpoint_every: Number of steps between checkpoints
        checkpoint_dtype: Data type for checkpoint storage (if None, uses original parameter dtype)
    """

    def __init__(
        self,
        checkpoint_dir: str | Path,
        max_checkpoints: int = 20,
        sigma_rels: tuple[float, ...] = (0.05, 0.28),
        update_every: int = 10,
        checkpoint_every: int = 1000,
        checkpoint_dtype: Optional[torch.dtype] = None,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_checkpoints = max_checkpoints
        self.checkpoint_dtype = checkpoint_dtype
        self.update_every = update_every
        self.checkpoint_every = checkpoint_every
        self.sigma_rels = sigma_rels
        self.gammas = tuple(map(sigma_rel_to_gamma, sigma_rels))
        
        self.step = 0
        self.ema_models = None

    @classmethod
    def from_model(
        cls,
        model: nn.Module,
        checkpoint_dir: str | Path,
        **kwargs,
    ) -> PostHocEMA:
        """
        Create PostHocEMA instance from a model for training.
        
        Args:
            model: Model to create EMAs from
            checkpoint_dir: Directory to store checkpoints
            **kwargs: Additional arguments passed to constructor
            
        Returns:
            PostHocEMA: Instance ready for training
        """
        instance = cls(checkpoint_dir=checkpoint_dir, **kwargs)
        instance.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        
        # Create CPU copy of model for EMA initialization
        cpu_model = deepcopy(model).cpu()
        
        # Initialize EMA models on CPU
        instance.ema_models = nn.ModuleList([
            KarrasEMA(
                cpu_model,
                sigma_rel=sigma_rel,
                update_every=instance.update_every,
            ) for sigma_rel in instance.sigma_rels
        ])
        
        # Clean up CPU model copy
        del cpu_model
        
        return instance

    @classmethod
    def from_path(
        cls,
        checkpoint_dir: str | Path,
        model: Optional[nn.Module] = None,
        **kwargs,
    ) -> PostHocEMA:
        """
        Load PostHocEMA instance from checkpoint directory.
        
        Args:
            checkpoint_dir: Directory containing checkpoints
            model: Optional model for parameter structure
            **kwargs: Additional arguments passed to constructor
            
        Returns:
            PostHocEMA: Instance ready for synthesis
        """
        checkpoint_dir = Path(checkpoint_dir)
        assert checkpoint_dir.exists(), f"Checkpoint directory {checkpoint_dir} does not exist"
        
        # Infer sigma_rels from checkpoint files if not provided
        if "sigma_rels" not in kwargs:
            # Find all unique indices in checkpoint files
            indices = set()
            for file in checkpoint_dir.glob("*.*.pt"):
                idx = int(file.stem.split(".")[0])
                indices.add(idx)
            
            # Sort indices to maintain order
            indices = sorted(indices)
            
            # Load first checkpoint for each index to get sigma_rel
            sigma_rels = []
            for idx in indices:
                checkpoint_file = next(checkpoint_dir.glob(f"{idx}.*.pt"))
                checkpoint = _safe_torch_load(str(checkpoint_file))
                sigma_rel = checkpoint.get("sigma_rel", None)
                if sigma_rel is not None:
                    sigma_rels.append(sigma_rel)
            
            if sigma_rels:
                kwargs["sigma_rels"] = tuple(sigma_rels)
        
        instance = cls(checkpoint_dir=checkpoint_dir, **kwargs)
        
        # Initialize EMA models if model provided
        if model is not None:
            instance.ema_models = nn.ModuleList([
                KarrasEMA(
                    model,
                    sigma_rel=sigma_rel,
                    update_every=instance.update_every,
                ) for sigma_rel in instance.sigma_rels
            ])
        
        return instance

    def update_(self, model: nn.Module) -> None:
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
            # Create checkpoint file
            checkpoint_file = self.checkpoint_dir / f"{idx}.{self.step}.pt"
            
            # Save EMA model state with sigma_rel
            state_dict = ema_model.state_dict()
            state_dict["sigma_rel"] = self.sigma_rels[idx]  # Add sigma_rel to checkpoint
            torch.save(state_dict, checkpoint_file)
            
            # Remove old checkpoints if needed
            checkpoint_files = sorted(
                self.checkpoint_dir.glob(f"{idx}.*.pt"),
                key=lambda p: int(p.stem.split(".")[1]),
            )
            if len(checkpoint_files) > self.max_checkpoints:
                for file in checkpoint_files[:-self.max_checkpoints]:
                    file.unlink()

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
        # Store original device and move base model to CPU
        original_device = next(base_model.parameters()).device
        base_model.cpu()
        torch.cuda.empty_cache()

        # Get state dict and create EMA model
        with self.state_dict(sigma_rel, step) as state_dict:
            ema_model = deepcopy(base_model)
            ema_model.load_state_dict(state_dict)

            try:
                yield ema_model
            finally:
                # Clean up EMA model and restore base model device
                if hasattr(ema_model, "cuda"):
                    ema_model.cpu()
                del ema_model
                base_model.to(original_device)
                torch.cuda.empty_cache()

    @contextmanager
    def state_dict(
        self,
        sigma_rel: float,
        step: int | None = None,
    ) -> Iterator[dict[str, torch.Tensor]]:
        """
        Context manager for getting state dict for synthesized EMA model.

        Args:
            sigma_rel: Target relative standard deviation
            step: Optional specific training step to synthesize for

        Yields:
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
        if self.ema_models is not None:
            # When we have ema_models, use their indices
            indices = range(len(self.ema_models))
        else:
            # When loading from path, find all unique indices
            indices = set()
            for file in self.checkpoint_dir.glob("*.*.pt"):
                idx = int(file.stem.split(".")[0])
                indices.add(idx)
            indices = sorted(indices)

        # Collect checkpoint info
        for idx in indices:
            checkpoint_files = sorted(
                self.checkpoint_dir.glob(f"{idx}.*.pt"),
                key=lambda p: int(p.stem.split(".")[1]),
            )
            for file in checkpoint_files:
                _, timestep = map(int, file.stem.split("."))
                # When we have ema_models, use their gammas
                if self.ema_models is not None:
                    gammas.append(self.gammas[idx])
                else:
                    # When loading from path, load gamma from checkpoint
                    checkpoint = _safe_torch_load(str(file))
                    sigma_rel = checkpoint.get("sigma_rel", None)
                    if sigma_rel is not None:
                        gammas.append(sigma_rel_to_gamma(sigma_rel))
                timesteps.append(timestep)
                checkpoints.append(file)

        # Use latest step if not specified
        step = step if step is not None else max(timesteps)
        assert step <= max(
            timesteps
        ), f"Cannot synthesize for step {step} > max available step {max(timesteps)}"

        # Solve for optimal weights using double precision
        gamma_i = torch.tensor(gammas, device=device, dtype=torch.float64)
        t_i = torch.tensor(timesteps, device=device, dtype=torch.float64)
        gamma_r = torch.tensor([gamma], device=device, dtype=torch.float64)
        t_r = torch.tensor([step], device=device, dtype=torch.float64)

        weights = self._solve_weights(t_i, gamma_i, t_r, gamma_r)
        weights = weights.squeeze(-1).to(dtype=torch.float64)  # Keep in float64

        # Load first checkpoint to get state dict structure
        ckpt = _safe_torch_load(str(checkpoints[0]), map_location=device)
        
        # Extract just the model parameters (remove EMA-specific keys)
        model_keys = {k.replace("ema_model.", ""): k for k in ckpt.keys() 
                     if k.startswith("ema_model.")}
        
        # Zero initialize synthesized state with double precision
        synth_state = {
            k: torch.zeros_like(ckpt[v], device=device, dtype=torch.float64) 
            for k, v in model_keys.items()
        }

        # Combine checkpoints using solved weights
        for checkpoint, weight in zip(checkpoints, weights.tolist()):
            ckpt_state = _safe_torch_load(str(checkpoint), map_location=device)
            for k, v in model_keys.items():
                # Convert checkpoint tensor to double precision
                ckpt_tensor = ckpt_state[v].to(dtype=torch.float64, device=device)
                # Use double precision for accumulation
                synth_state[k].add_(ckpt_tensor * weight)

        # Convert final state to target dtype
        synth_state = {
            k: v.to(dtype=self.checkpoint_dtype if self.checkpoint_dtype is not None else v.dtype, device=device)
            for k, v in synth_state.items()
        }

        try:
            yield synth_state
        finally:
            # Clean up tensors
            del synth_state
            torch.cuda.empty_cache()

    def _solve_weights(
        self,
        t_i: torch.Tensor,
        gamma_i: torch.Tensor,
        t_r: torch.Tensor,
        gamma_r: torch.Tensor,
    ) -> torch.Tensor:
        """
        Solve for optimal weights to synthesize target EMA profile.
        
        Args:
            t_i: Timesteps of stored checkpoints
            gamma_i: Gamma values of stored checkpoints
            t_r: Target timestep
            gamma_r: Target gamma value
            
        Returns:
            torch.Tensor: Optimal weights for combining checkpoints
        """
        return solve_weights(t_i, gamma_i, t_r, gamma_r)
