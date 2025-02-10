"""Core EMA functionality adapted from lucidrains implementation."""

from __future__ import annotations

from copy import deepcopy
from functools import partial
from typing import Callable

import torch
from torch import Tensor, nn
from torch.nn import Module

from .utils import exists, sigma_rel_to_gamma


def get_module_device(m: Module):
    """Get the device of a PyTorch module by checking its first parameter."""
    return next(m.parameters()).device


def inplace_copy(tgt: Tensor, src: Tensor):
    """
    Inplace copy of src tensor to tgt tensor.

    Args:
        tgt: Target tensor to copy to
        src: Source tensor to copy from
    """
    tgt.copy_(src.to(tgt.device))


def inplace_lerp(tgt: Tensor, src: Tensor, weight):
    """
    Inplace linear interpolation between tgt and src tensors.

    Args:
        tgt: Target tensor to interpolate
        src: Source tensor to interpolate towards
        weight: Interpolation weight between 0 and 1
    """
    tgt.lerp_(src.to(tgt.device), weight)


class KarrasEMA(Module):
    """
    Karras EMA implementation with power function decay profile.

    Args:
        model: Model to create EMA of
        sigma_rel: Relative standard deviation for EMA profile
        gamma: Alternative parameterization via gamma (don't specify both)
        ema_model: Optional pre-initialized EMA model
        update_every: Number of steps between EMA updates
        frozen: Whether to freeze EMA updates
        param_or_buffer_names_no_ema: Parameter/buffer names to exclude from EMA
        ignore_names: Parameter/buffer names to ignore
        ignore_startswith_names: Parameter/buffer name prefixes to ignore
        only_save_diff: If True, only save parameters with requires_grad=True
    """

    def __init__(
        self,
        model: Module,
        sigma_rel: float | None = None,
        gamma: float | None = None,
        ema_model: Module | Callable[[], Module] | None = None,
        update_every: int = 10,
        frozen: bool = False,
        param_or_buffer_names_no_ema: set[str] = set(),
        ignore_names: set[str] = set(),
        ignore_startswith_names: set[str] = set(),
        only_save_diff: bool = False,
    ):
        super().__init__()

        assert exists(sigma_rel) ^ exists(
            gamma
        ), "either sigma_rel or gamma must be given"

        if exists(sigma_rel):
            gamma = sigma_rel_to_gamma(sigma_rel)

        self.gamma = gamma
        self.frozen = frozen
        self.update_every = update_every
        self.only_save_diff = only_save_diff

        # Store reference to online model
        self.online_model = [model]

        # Initialize EMA model
        if callable(ema_model) and not isinstance(ema_model, Module):
            ema_model = ema_model()

        # Store original device
        original_device = next(model.parameters()).device

        # Move model to CPU before copying to avoid VRAM spike
        model.cpu()

        try:
            # Create EMA model on CPU
            self.ema_model = (ema_model if exists(ema_model) else deepcopy(model)).cpu()

            # Ensure all parameters and buffers are on CPU and detached
            for p in self.ema_model.parameters():
                p.data = p.data.cpu().detach()
            for b in self.ema_model.buffers():
                b.data = b.data.cpu().detach()

            # Move model back to original device
            model.to(original_device)

            # Get parameter names for floating point or complex parameters
            self.param_names = {
                name
                for name, param in self.ema_model.named_parameters()
                if torch.is_floating_point(param) or torch.is_complex(param)
            }

            # Get buffer names for floating point or complex buffers
            self.buffer_names = {
                name
                for name, buffer in self.ema_model.named_buffers()
                if torch.is_floating_point(buffer) or torch.is_complex(buffer)
            }

            # Names to ignore
            self.param_or_buffer_names_no_ema = param_or_buffer_names_no_ema
            self.ignore_names = ignore_names
            self.ignore_startswith_names = ignore_startswith_names

            # State buffers on CPU
            self.register_buffer("initted", torch.tensor(False, device="cpu"))
            self.register_buffer("step", torch.tensor(0, device="cpu"))
        except:
            # Ensure model is moved back even if initialization fails
            model.to(original_device)
            raise

    @property
    def beta(self):
        """Calculate current beta value for EMA update."""
        return (1.0 - 1.0 / (self.step.item() + 1.0)) ** (1.0 + self.gamma)

    def update(self):
        """Update EMA weights if conditions are met."""
        step = self.step.item()
        self.step += 1

        if step % self.update_every != 0:
            return

        if not self.initted.item():
            self.copy_params_from_model_to_ema()
            self.initted.data.copy_(torch.tensor(True, device=self.initted.device))

        if not self.frozen:
            self.update_moving_average()

    def copy_params_from_model_to_ema(self):
        """Copy parameters from online model to EMA model."""
        # Copy parameters
        for (name, ma_params), (_, current_params) in zip(
            self.get_params_iter(self.ema_model),
            self.get_params_iter(self.online_model[0]),
        ):
            if self._should_update_param(name):
                inplace_copy(ma_params.data, current_params.data)

        # Copy buffers
        for (name, ma_buffer), (_, current_buffer) in zip(
            self.get_buffers_iter(self.ema_model),
            self.get_buffers_iter(self.online_model[0]),
        ):
            if self._should_update_param(name):
                inplace_copy(ma_buffer.data, current_buffer.data)

    def update_moving_average(self):
        """Update EMA weights using current beta value."""
        current_decay = self.beta

        # Update parameters
        for (name, current_params), (_, ma_params) in zip(
            self.get_params_iter(self.online_model[0]),
            self.get_params_iter(self.ema_model),
        ):
            if not self._should_update_param(name):
                continue
            inplace_lerp(ma_params.data, current_params.data, 1.0 - current_decay)

        # Update buffers
        for (name, current_buffer), (_, ma_buffer) in zip(
            self.get_buffers_iter(self.online_model[0]),
            self.get_buffers_iter(self.ema_model),
        ):
            if not self._should_update_param(name):
                continue
            inplace_lerp(ma_buffer.data, current_buffer.data, 1.0 - current_decay)

    def _should_update_param(self, name: str) -> bool:
        """Check if parameter should be updated based on ignore rules."""
        if name in self.ignore_names:
            return False
        if any(name.startswith(prefix) for prefix in self.ignore_startswith_names):
            return False
        if name in self.param_or_buffer_names_no_ema:
            return False
        return True

    def get_params_iter(self, model):
        """Get iterator over model's parameters."""
        for name, param in model.named_parameters():
            if name not in self.param_names:
                continue
            if self.only_save_diff and not param.requires_grad:
                continue
            yield name, param

    def get_buffers_iter(self, model):
        """Get iterator over model's buffers."""
        for name, buffer in model.named_buffers():
            if name not in self.buffer_names:
                continue
            if self.only_save_diff and not buffer.requires_grad:
                continue
            yield name, buffer

    def iter_all_ema_params_and_buffers(self):
        """Get iterator over all EMA parameters and buffers."""
        for name, param in self.ema_model.named_parameters():
            if name not in self.param_names:
                continue
            if name in self.param_or_buffer_names_no_ema:
                continue
            if name in self.ignore_names:
                continue
            if any(name.startswith(prefix) for prefix in self.ignore_startswith_names):
                continue
            yield param

        for name, buffer in self.ema_model.named_buffers():
            if name not in self.buffer_names:
                continue
            if name in self.param_or_buffer_names_no_ema:
                continue
            if name in self.ignore_names:
                continue
            if any(name.startswith(prefix) for prefix in self.ignore_startswith_names):
                continue
            yield buffer

    def iter_all_model_params_and_buffers(self, model: Module):
        """Get iterator over all model parameters and buffers."""
        for name, param in model.named_parameters():
            if name not in self.param_names:
                continue
            if name in self.param_or_buffer_names_no_ema:
                continue
            if name in self.ignore_names:
                continue
            if any(name.startswith(prefix) for prefix in self.ignore_startswith_names):
                continue
            yield param

        for name, buffer in model.named_buffers():
            if name not in self.buffer_names:
                continue
            if name in self.param_or_buffer_names_no_ema:
                continue
            if name in self.ignore_names:
                continue
            if any(name.startswith(prefix) for prefix in self.ignore_startswith_names):
                continue
            yield buffer

    def __call__(self, *args, **kwargs):
        """Forward pass using EMA model."""
        return self.ema_model(*args, **kwargs)

    def state_dict(self):
        """Get state dict for EMA model."""
        state_dict = {}

        # Save parameters based on only_save_diff flag
        for name, param in self.ema_model.named_parameters():
            if name not in self.param_names:
                continue
            if self.only_save_diff and not param.requires_grad:
                continue
            state_dict[name] = param.data

        # Save buffers
        for name, buffer in self.ema_model.named_buffers():
            if name not in self.buffer_names:
                continue
            state_dict[name] = buffer.data

        # Save internal state
        state_dict["initted"] = self.initted.data
        state_dict["step"] = self.step.data

        return state_dict

    def load_state_dict(self, state_dict):
        """Load state dict into EMA model."""
        # Load parameters based on only_save_diff flag
        for name, param in self.ema_model.named_parameters():
            if (not self.only_save_diff or param.requires_grad) and name in state_dict:
                param.data.copy_(state_dict[name].data)

        # Load buffers
        for name, buffer in self.ema_model.named_buffers():
            if name in state_dict:
                buffer.data.copy_(state_dict[name].data)

        # Load internal state
        if "initted" in state_dict:
            self.initted.data.copy_(state_dict["initted"].data)
        if "step" in state_dict:
            self.step.data.copy_(state_dict["step"].data)
