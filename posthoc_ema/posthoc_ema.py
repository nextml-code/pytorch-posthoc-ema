from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Iterator, Optional, Generator, Dict

import torch
from PIL import Image
from torch import nn
import pickle
import io
import torch.serialization

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
        calculation_dtype: Data type for synthesis calculations (default=torch.float32)
        only_save_diff: If True, only save parameters with requires_grad=True
        update_after_step: Number of steps after which to update EMA models
    """

    def __init__(
        self,
        checkpoint_dir: str | Path,
        max_checkpoints: int = 20,
        sigma_rels: tuple[float, ...] | None = None,
        update_every: int = 10,
        checkpoint_every: int = 1000,
        checkpoint_dtype: Optional[torch.dtype] = None,
        calculation_dtype: torch.dtype = torch.float32,
        only_save_diff: bool = False,
        update_after_step: int = 100,
    ):
        if sigma_rels is None:
            sigma_rels = (0.05, 0.28)  # Default values from paper

        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_checkpoints = max_checkpoints
        self.checkpoint_dtype = checkpoint_dtype
        self.calculation_dtype = calculation_dtype
        self.update_every = update_every
        self.checkpoint_every = checkpoint_every
        self.only_save_diff = only_save_diff
        self.update_after_step = update_after_step

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
        calculation_dtype: torch.dtype = torch.float32,
        only_save_diff: bool = False,
        update_after_step: int = 100,
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
            calculation_dtype: Data type for synthesis calculations (default=torch.float32)
            only_save_diff: If True, only save parameters with requires_grad=True
            update_after_step: Number of steps after which to update EMA models

        Returns:
            PostHocEMA: Instance ready for training

        Raises:
            ValueError: If checkpoint directory already exists and contains checkpoints
        """
        checkpoint_dir = Path(checkpoint_dir)
        if checkpoint_dir.exists():
            checkpoints = list(checkpoint_dir.glob("*.pt"))
            if checkpoints:
                raise ValueError(
                    f"Checkpoint directory {checkpoint_dir} already contains checkpoints. "
                    "Use from_path() to load existing checkpoints instead of from_model()."
                )

        instance = cls(
            checkpoint_dir=checkpoint_dir,
            max_checkpoints=max_checkpoints,
            sigma_rels=sigma_rels,
            update_every=update_every,
            checkpoint_every=checkpoint_every,
            checkpoint_dtype=checkpoint_dtype,
            calculation_dtype=calculation_dtype,
            only_save_diff=only_save_diff,
            update_after_step=update_after_step,
        )
        instance.checkpoint_dir.mkdir(exist_ok=True, parents=True)

        # Store original device
        original_device = next(model.parameters()).device

        # Move model to CPU before copying to avoid VRAM spike
        model.cpu()

        try:
            # Initialize EMA models on CPU
            instance.ema_models = nn.ModuleList(
                [
                    KarrasEMA(
                        model,
                        sigma_rel=sigma_rel,
                        update_every=instance.update_every,
                        only_save_diff=instance.only_save_diff,
                    )
                    for sigma_rel in instance.sigma_rels
                ]
            )

            # Move model back to original device
            model.to(original_device)

            return instance
        except:
            # Ensure model is moved back even if initialization fails
            model.to(original_device)
            raise

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
        only_save_diff: bool = False,
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
            only_save_diff: If True, only save parameters with requires_grad=True

        Returns:
            PostHocEMA: Instance ready for synthesis
        """
        checkpoint_dir = Path(checkpoint_dir)
        assert (
            checkpoint_dir.exists()
        ), f"Checkpoint directory {checkpoint_dir} does not exist"

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
            only_save_diff=only_save_diff,
        )

        # Initialize EMA models if model provided
        if model is not None:
            instance.ema_models = nn.ModuleList(
                [
                    KarrasEMA(
                        model,
                        sigma_rel=sigma_rel,
                        update_every=instance.update_every,
                        only_save_diff=instance.only_save_diff,
                    )
                    for sigma_rel in instance.sigma_rels
                ]
            )

        return instance

    def update_(self, model: nn.Module) -> None:
        """
        Update EMA models and create checkpoints if needed.

        Args:
            model: Current state of the model to update EMAs with
        """
        self.step += 1

        # Only update after update_after_step steps
        if self.step < self.update_after_step:
            return

        # Update EMA models with current model state
        for ema_model in self.ema_models:
            # Update online model reference and copy parameters
            ema_model.online_model[0] = model
            if not ema_model.initted.item():
                ema_model.copy_params_from_model_to_ema()
                ema_model.initted.data.copy_(torch.tensor(True))
            ema_model.update()

        # Create checkpoint if needed
        if self.step % self.checkpoint_every == 0:
            self._create_checkpoint()
            self._cleanup_old_checkpoints()

    def _create_checkpoint(self) -> None:
        """Create checkpoints for all EMA models."""
        for idx, ema_model in enumerate(self.ema_models):
            # Create checkpoint file
            checkpoint_file = self.checkpoint_dir / f"{idx}.{self.step}.pt"

            # Get state dict from EMA model
            state_dict = ema_model.state_dict()

            # Filter parameters based on only_save_diff
            if self.only_save_diff:
                filtered_state_dict = {}
                for name, param in ema_model.ema_model.named_parameters():
                    if param.requires_grad:
                        key = name
                        if key in state_dict:
                            filtered_state_dict[key] = state_dict[key]
                # Add buffers and internal state
                for name, buffer in ema_model.ema_model.named_buffers():
                    key = name
                    if key in state_dict:
                        filtered_state_dict[key] = state_dict[key]
                for key in ["initted", "step"]:
                    if key in state_dict:
                        filtered_state_dict[key] = state_dict[key]
                state_dict = filtered_state_dict

            # Convert to checkpoint dtype if specified
            if self.checkpoint_dtype is not None:
                state_dict = {
                    k: v.to(self.checkpoint_dtype) if isinstance(v, torch.Tensor) else v
                    for k, v in state_dict.items()
                }

            # Save checkpoint
            torch.save(state_dict, checkpoint_file)

            # Remove old checkpoints if needed
            checkpoint_files = sorted(
                self.checkpoint_dir.glob(f"{idx}.*.pt"),
                key=lambda p: int(p.stem.split(".")[1]),
            )
            if len(checkpoint_files) > self.max_checkpoints:
                for file in checkpoint_files[: -self.max_checkpoints]:
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
        model: nn.Module,
        sigma_rel: float,
        *,
        calculation_dtype: torch.dtype = torch.float32,
    ) -> Iterator[nn.Module]:
        """
        Context manager for temporarily setting model parameters to EMA state.

        Args:
            model: Model to temporarily set to EMA state
            sigma_rel: Target relative standard deviation
            calculation_dtype: Data type for synthesis calculations (default=torch.float32)

        Yields:
            nn.Module: Model with EMA parameters
        """
        # Move model to CPU for memory efficiency
        original_device = next(model.parameters()).device
        model.cpu()
        torch.cuda.empty_cache()

        try:
            with self.state_dict(
                sigma_rel, calculation_dtype=calculation_dtype
            ) as state_dict:
                # Store original state only for parameters that will be modified
                original_state = {
                    name: param.detach().clone()
                    for name, param in model.state_dict().items()
                    if name in state_dict
                }

                # Load EMA state directly into model
                result = model.load_state_dict(
                    state_dict, strict=not self.only_save_diff
                )
                assert (
                    len(result.unexpected_keys) == 0
                ), f"Unexpected keys: {result.unexpected_keys}"
                model.eval()  # Set to eval mode to handle BatchNorm
                yield model

                # Restore original state
                model.load_state_dict(original_state, strict=False)
                del original_state
                del state_dict  # Free memory for state dict
                torch.cuda.empty_cache()
        finally:
            # Restore model to original device
            model.to(original_device)
            torch.cuda.empty_cache()

    @contextmanager
    def state_dict(
        self,
        sigma_rel: float,
        *,
        calculation_dtype: torch.dtype = torch.float32,
    ) -> Iterator[Dict[str, torch.Tensor]]:
        """
        Context manager for getting state dict for synthesized EMA model.

        Args:
            sigma_rel: Target relative standard deviation
            calculation_dtype: Data type for synthesis calculations (default=torch.float32)

        Yields:
            dict[str, torch.Tensor]: State dict with synthesized weights
        """
        # Convert target sigma_rel to gamma
        gamma = sigma_rel_to_gamma(sigma_rel)
        device = torch.device("cpu")  # Keep synthesis on CPU for memory efficiency

        # First count total checkpoints to pre-allocate tensors
        total_checkpoints = 0
        checkpoint_files = []
        if self.ema_models is not None:
            # When we have ema_models, use their indices
            for idx in range(len(self.ema_models)):
                files = sorted(
                    self.checkpoint_dir.glob(f"{idx}.*.pt"),
                    key=lambda p: int(p.stem.split(".")[1]),
                )
                total_checkpoints += len(files)
                checkpoint_files.extend(files)
        else:
            # When loading from path, find all unique indices
            indices = set()
            for file in self.checkpoint_dir.glob("*.*.pt"):
                idx = int(file.stem.split(".")[0])
                indices.add(idx)
            indices = sorted(indices)

            for idx in indices:
                files = sorted(
                    self.checkpoint_dir.glob(f"{idx}.*.pt"),
                    key=lambda p: int(p.stem.split(".")[1]),
                )
                total_checkpoints += len(files)
                checkpoint_files.extend(files)

        if total_checkpoints == 0:
            raise ValueError("No checkpoints found")

        # Pre-allocate tensors in calculation dtype
        gammas = torch.empty(total_checkpoints, dtype=calculation_dtype, device=device)
        timesteps = torch.empty(
            total_checkpoints, dtype=calculation_dtype, device=device
        )

        # Fill tensors one value at a time
        for i, file in enumerate(checkpoint_files):
            idx = int(file.stem.split(".")[0])
            timestep = int(file.stem.split(".")[1])
            timesteps[i] = float(timestep)  # Convert to float

            if self.ema_models is not None:
                gammas[i] = self.gammas[idx]
            else:
                # Load gamma from checkpoint
                checkpoint = torch.load(
                    str(file), weights_only=True, map_location="cpu"
                )
                sigma_rel = checkpoint.get("sigma_rel", None)
                if sigma_rel is not None:
                    gammas[i] = sigma_rel_to_gamma(sigma_rel)
                else:
                    gammas[i] = self.gammas[idx]
                del checkpoint  # Free memory immediately
                torch.cuda.empty_cache()

        # Solve for weights in calculation dtype
        weights = solve_weights(
            gammas,
            timesteps,
            gamma,
            calculation_dtype=calculation_dtype,
            target_sigma_rel=sigma_rel,
        )

        # Free memory for gamma and timestep tensors
        del gammas
        del timesteps
        torch.cuda.empty_cache()

        # Load first checkpoint to get parameter names and original dtypes
        first_checkpoint = torch.load(
            str(checkpoint_files[0]), weights_only=True, map_location="cpu"
        )
        param_names = {
            k: k for k in first_checkpoint.keys() if k not in ("initted", "step")
        }
        # Store original dtypes for each parameter
        param_dtypes = {
            name: first_checkpoint[checkpoint_name].dtype
            for name, checkpoint_name in param_names.items()
            if isinstance(first_checkpoint[checkpoint_name], torch.Tensor)
        }
        del first_checkpoint
        torch.cuda.empty_cache()

        # Initialize state dict with empty tensors
        state_dict = {}

        # Process one checkpoint at a time
        for file_idx, (file, weight) in enumerate(zip(checkpoint_files, weights)):
            # Load checkpoint
            checkpoint = torch.load(str(file), weights_only=True, map_location="cpu")

            # Process all parameters from this checkpoint
            for param_name, checkpoint_name in param_names.items():
                if checkpoint_name not in checkpoint:
                    # If parameter is missing from checkpoint but we're not in only_save_diff mode,
                    # or if it's a parameter with requires_grad=True, this is an error
                    if not self.only_save_diff:
                        raise ValueError(
                            f"Parameter {param_name} missing from checkpoint {file} "
                            "but only_save_diff=False"
                        )
                    # Skip parameters that are intentionally not saved in only_save_diff mode
                    continue

                param_data = checkpoint[checkpoint_name]
                if not isinstance(param_data, torch.Tensor):
                    continue

                # Convert to calculation dtype for synthesis
                param_data = param_data.to(calculation_dtype)

                if file_idx == 0:
                    # Initialize parameter with first weighted contribution
                    state_dict[param_name] = param_data.to(device) * weight
                else:
                    # Add weighted contribution to existing parameter
                    state_dict[param_name].add_(param_data.to(device) * weight)

            # Free memory for this checkpoint
            del checkpoint
            torch.cuda.empty_cache()

        # Convert back to original dtypes
        for name, tensor in state_dict.items():
            if name in param_dtypes:
                state_dict[name] = tensor.to(param_dtypes[name])

        # Free memory
        del weights
        torch.cuda.empty_cache()

        try:
            yield state_dict
        finally:
            # Clean up
            del state_dict
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
