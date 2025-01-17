from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Iterator, Optional

import torch
from PIL import Image
from torch import nn

from .karras_ema import KarrasEMA
from .utils import _safe_torch_load, p_dot_p, sigma_rel_to_gamma, solve_weights
from .visualization import compute_reconstruction_errors, plot_reconstruction_errors


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
        sigma_rels: tuple[float, ...] | None = None,
        update_every: int = 10,
        checkpoint_every: int = 1000,
        checkpoint_dtype: Optional[torch.dtype] = None,
    ):
        if sigma_rels is None:
            sigma_rels = (0.05, 0.28)  # Default values from paper

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
        max_checkpoints: int = 20,
        sigma_rels: tuple[float, ...] | None = None,
        update_every: int = 10,
        checkpoint_every: int = 1000,
        checkpoint_dtype: Optional[torch.dtype] = None,
    ) -> PostHocEMA:
        """
        Create PostHocEMA instance from a model for training.
        
        Args:
            model: Model to create EMAs from
            checkpoint_dir: Directory to store checkpoints
            max_checkpoints: Maximum number of checkpoints to keep per EMA model
            sigma_rels: Tuple of relative standard deviations for the maintained EMA models
            update_every: Number of steps between EMA updates
            checkpoint_every: Number of steps between checkpoints
            checkpoint_dtype: Data type for checkpoint storage (if None, uses original parameter dtype)
            
        Returns:
            PostHocEMA: Instance ready for training
        """
        instance = cls(
            checkpoint_dir=checkpoint_dir,
            max_checkpoints=max_checkpoints,
            sigma_rels=sigma_rels,
            update_every=update_every,
            checkpoint_every=checkpoint_every,
            checkpoint_dtype=checkpoint_dtype,
        )
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
        max_checkpoints: int = 20,
        sigma_rels: tuple[float, ...] | None = None,
        update_every: int = 10,
        checkpoint_every: int = 1000,
        checkpoint_dtype: Optional[torch.dtype] = None,
    ) -> PostHocEMA:
        """
        Load PostHocEMA instance from checkpoint directory.
        
        Args:
            checkpoint_dir: Directory containing checkpoints
            model: Optional model for parameter structure
            max_checkpoints: Maximum number of checkpoints to keep per EMA model
            sigma_rels: Tuple of relative standard deviations for the maintained EMA models
            update_every: Number of steps between EMA updates
            checkpoint_every: Number of steps between checkpoints
            checkpoint_dtype: Data type for checkpoint storage (if None, uses original parameter dtype)
            
        Returns:
            PostHocEMA: Instance ready for synthesis
        """
        checkpoint_dir = Path(checkpoint_dir)
        assert checkpoint_dir.exists(), f"Checkpoint directory {checkpoint_dir} does not exist"
        
        # Infer sigma_rels from checkpoint files if not provided
        if sigma_rels is None:
            # Find all unique indices in checkpoint files
            indices = set()
            for file in checkpoint_dir.glob("*.*.pt"):
                idx = int(file.stem.split(".")[0])
                indices.add(idx)
            
            # Sort indices to maintain order
            indices = sorted(indices)
            
            # Load first checkpoint for each index to get sigma_rel
            sigma_rels_list = []
            for idx in indices:
                checkpoint_file = next(checkpoint_dir.glob(f"{idx}.*.pt"))
                checkpoint = _safe_torch_load(str(checkpoint_file))
                sigma_rel = checkpoint.get("sigma_rel", None)
                if sigma_rel is not None:
                    sigma_rels_list.append(sigma_rel)
            
            if sigma_rels_list:
                sigma_rels = tuple(sigma_rels_list)
        
        instance = cls(
            checkpoint_dir=checkpoint_dir,
            max_checkpoints=max_checkpoints,
            sigma_rels=sigma_rels,
            update_every=update_every,
            checkpoint_every=checkpoint_every,
            checkpoint_dtype=checkpoint_dtype,
        )
        
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
            
            # Save EMA model state with correct dtype and ema_model prefix
            state_dict = {
                f"ema_model.{k}": v.to(self.checkpoint_dtype) if self.checkpoint_dtype is not None else v
                for k, v in ema_model.state_dict().items()
            }
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
        with self.state_dict(sigma_rel=sigma_rel, step=step) as state_dict:
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
                    else:
                        # If no sigma_rel in checkpoint, use index-based gamma
                        gammas.append(self.gammas[idx])
                timesteps.append(timestep)
                checkpoints.append(file)

        if not gammas:
            raise ValueError("No valid gamma values found in checkpoints")

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
        
        # Extract model parameters, handling both formats and filtering out internal state
        model_keys = {}
        for k in ckpt.keys():
            # Skip internal EMA tracking variables
            if k in ("initted", "step", "sigma_rel"):
                continue
            if k.startswith("ema_model."):
                # Reference format: "ema_model.weight" -> "weight"
                model_keys[k] = k.replace("ema_model.", "")
            else:
                # Our format: "weight" -> "weight"
                model_keys[k] = k

        # Zero initialize synthesized state with double precision
        synth_state = {}
        for ref_key, our_key in model_keys.items():
            if ref_key in ckpt:
                synth_state[our_key] = torch.zeros_like(ckpt[ref_key], device=device, dtype=torch.float64)
            elif our_key in ckpt:
                synth_state[our_key] = torch.zeros_like(ckpt[our_key], device=device, dtype=torch.float64)

        # Combine checkpoints using solved weights
        for checkpoint, weight in zip(checkpoints, weights.tolist()):
            ckpt_state = _safe_torch_load(str(checkpoint), map_location=device)
            for ref_key, our_key in model_keys.items():
                if ref_key in ckpt_state:
                    ckpt_tensor = ckpt_state[ref_key]
                elif our_key in ckpt_state:
                    ckpt_tensor = ckpt_state[our_key]
                else:
                    continue
                # Convert checkpoint tensor to double precision
                ckpt_tensor = ckpt_tensor.to(dtype=torch.float64, device=device)
                # Use double precision for accumulation
                synth_state[our_key].add_(ckpt_tensor * weight)

        # Convert final state to target dtype and filter out internal state
        synth_state = {
            k: v.to(dtype=self.checkpoint_dtype if self.checkpoint_dtype is not None else v.dtype, device=device)
            for k, v in synth_state.items()
            if k not in ("initted", "step", "sigma_rel")  # Filter out internal state
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

    def reconstruction_error(
        self,
        target_sigma_rel_range: tuple[float, float] | None = None,
    ) -> Image.Image:
        """
        Generate a plot showing reconstruction errors for different target sigma_rel values.
        
        This shows how well we can reconstruct different EMA profiles using our stored checkpoints.
        Lower error indicates better reconstruction. The error should be minimal around the source
        sigma_rel values, as these profiles can be reconstructed exactly.
        
        Args:
            target_sigma_rel_range: Range of sigma_rel values to test (min, max).
                                  Defaults to (0.05, 0.28) which covers common values.
            
        Returns:
            PIL.Image.Image: Plot showing reconstruction errors for different sigma_rel values
        """
        target_sigma_rels, errors, _ = compute_reconstruction_errors(
            sigma_rels=self.sigma_rels,
            target_sigma_rel_range=target_sigma_rel_range,
        )
        
        return plot_reconstruction_errors(
            target_sigma_rels=target_sigma_rels,
            errors=errors,
            source_sigma_rels=self.sigma_rels,
        )
