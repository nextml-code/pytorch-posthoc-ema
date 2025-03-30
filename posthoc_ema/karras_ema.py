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
    # Check if tensor is integer type - integer tensors can't use lerp
    # but we want to silently handle them instead of raising errors
    if tgt.dtype in [torch.int, torch.int8, torch.int16, torch.int32, torch.int64, torch.long]:
        tgt.copy_(src.to(tgt.device))
    else:
        tgt.lerp_(src.to(tgt.device), weight)


class KarrasEMA(Module):
    """
    Karras EMA implementation with power function decay profile.

    Args:
        model: Model to create EMA of
        sigma_rel: Relative standard deviation for EMA profile
        gamma: Alternative parameterization via gamma (don't specify both)
        update_every: Number of steps between EMA updates
        frozen: Whether to freeze EMA updates
        param_or_buffer_names_no_ema: Parameter/buffer names to exclude from EMA
        ignore_names: Parameter/buffer names to ignore
        ignore_startswith_names: Parameter/buffer name prefixes to ignore
        only_save_diff: If True, only save parameters with requires_grad=True
        device: Device to store EMA parameters on (default='cpu')
    """

    # Buffers that should always be included in the state dict even with only_save_diff=True
    _ALWAYS_INCLUDE_BUFFERS = {"running_mean", "running_var", "num_batches_tracked"}
    
    def __init__(
        self,
        model: Module,
        sigma_rel: float | None = None,
        gamma: float | None = None,
        update_every: int = 10,
        frozen: bool = False,
        param_or_buffer_names_no_ema: set[str] = set(),
        ignore_names: set[str] = set(),
        ignore_startswith_names: set[str] = set(),
        only_save_diff: bool = False,
        device: str = 'cpu',
    ):
        super().__init__()
        
        # Store all the configuration parameters first
        self.gamma = gamma
        self.frozen = frozen
        self.update_every = update_every
        self.only_save_diff = only_save_diff
        self.ignore_names = ignore_names
        self.ignore_startswith_names = ignore_startswith_names
        self.param_or_buffer_names_no_ema = param_or_buffer_names_no_ema
        self.device = device
        
        assert exists(sigma_rel) ^ exists(gamma), "either sigma_rel or gamma must be given"
        
        if exists(sigma_rel):
            gamma = sigma_rel_to_gamma(sigma_rel)
            self.gamma = gamma
        
        # Store reference to online model
        self.online_model = [model]
        
        # Instead of copying the whole model, just store parameter tensors
        self.ema_params = {}
        self.ema_buffers = {}
        
        # Get parameter and buffer names to track
        with torch.no_grad():
            for name, param in model.named_parameters():
                if self._should_update_param(name):
                    if not only_save_diff or param.requires_grad:
                        self.ema_params[name] = param.detach().clone().to(self.device)
                        
            for name, buffer in model.named_buffers():
                if self._should_update_param(name):
                    buffer_name = name.split('.')[-1]  # Get the base name
                    # Always include critical buffers regardless of only_save_diff
                    if not only_save_diff or buffer.requires_grad or buffer_name in self._ALWAYS_INCLUDE_BUFFERS:
                        self.ema_buffers[name] = buffer.detach().clone().to(self.device)
        
        # State buffers
        self.register_buffer("initted", torch.tensor(False))
        self.register_buffer("step", torch.tensor(0))

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
        with torch.no_grad():
            for name, param in self.online_model[0].named_parameters():
                if name in self.ema_params:
                    # Explicitly move to device (usually CPU)
                    self.ema_params[name] = param.detach().clone().to(self.device)

            # Copy buffers
            for name, buffer in self.online_model[0].named_buffers():
                if name in self.ema_buffers:
                    # Explicitly move to device (usually CPU)
                    self.ema_buffers[name] = buffer.detach().clone().to(self.device)

    def update_moving_average(self):
        """Update EMA weights using current beta value."""
        current_decay = self.beta

        # Update parameters using the simplified lerp function (which now handles integer tensors)
        for name, current_params in self.online_model[0].named_parameters():
            if name in self.ema_params:
                # inplace_lerp now handles integer tensors internally
                inplace_lerp(self.ema_params[name], current_params.data, 1.0 - current_decay)

        # Update buffers with the same simplified approach
        for name, current_buffer in self.online_model[0].named_buffers():
            if name in self.ema_buffers:
                # inplace_lerp now handles integer tensors internally
                inplace_lerp(self.ema_buffers[name], current_buffer.data, 1.0 - current_decay)

    def _should_update_param(self, name: str) -> bool:
        """Check if parameter should be updated based on ignore rules."""
        if name in self.ignore_names:
            return False
        if any(name.startswith(prefix) for prefix in self.ignore_startswith_names):
            return False
        if name in self.param_or_buffer_names_no_ema:
            return False
        return True

    def _parameter_requires_grad(self, name: str) -> bool:
        """Check if parameter requires gradients in the online model."""
        for n, p in self.online_model[0].named_parameters():
            if n == name:
                return p.requires_grad
        return False

    def get_params_iter(self, model):
        """Get iterator over model's parameters."""
        for name, param in model.named_parameters():
            if name not in self.ema_params:
                continue
            if self.only_save_diff and not param.requires_grad:
                continue
            yield name, param

    def get_buffers_iter(self, model):
        """Get iterator over model's buffers."""
        for name, buffer in model.named_buffers():
            if name not in self.ema_buffers:
                continue
            
            # Handle critical buffers that should always be included
            buffer_name = name.split('.')[-1]
            if self.only_save_diff and not buffer.requires_grad and buffer_name not in self._ALWAYS_INCLUDE_BUFFERS:
                continue
            
            yield name, buffer

    def iter_all_ema_params_and_buffers(self):
        """Get iterator over all EMA parameters and buffers."""
        for name, param in self.ema_params.items():
            if name in self.param_or_buffer_names_no_ema:
                continue
            if name in self.ignore_names:
                continue
            if any(name.startswith(prefix) for prefix in self.ignore_startswith_names):
                continue
            yield param

    def iter_all_model_params_and_buffers(self, model: Module):
        """Get iterator over all model parameters and buffers."""
        for name, param in model.named_parameters():
            if name not in self.ema_params:
                continue
            if name in self.param_or_buffer_names_no_ema:
                continue
            if name in self.ignore_names:
                continue
            if any(name.startswith(prefix) for prefix in self.ignore_startswith_names):
                continue
            yield param

    def __call__(self, *args, **kwargs):
        """Forward pass using EMA model."""
        raise NotImplementedError("KarrasEMA no longer maintains a full model copy")

    @property
    def ema_model(self):
        """
        For backward compatibility with tests.
        Creates a temporary model with EMA parameters.
        
        Returns:
            Module: A copy of the online model with EMA parameters
        """
        # Create a copy of the online model
        model_copy = deepcopy(self.online_model[0])
        
        # Load EMA parameters into the model
        for name, param in model_copy.named_parameters():
            if name in self.ema_params:
                param.data.copy_(self.ema_params[name])
            
        # Load EMA buffers into the model
        for name, buffer in model_copy.named_buffers():
            if name in self.ema_buffers:
                buffer.data.copy_(self.ema_buffers[name])
        
        # Ensure the model is on CPU
        model_copy.to('cpu')
        return model_copy

    def state_dict(self):
        """Get state dict with EMA parameters."""
        state_dict = {}
        
        # For parameters, respect only_save_diff
        for name, param in self.ema_params.items():
            if not self.only_save_diff or self._parameter_requires_grad(name):
                state_dict[name] = param.data
        
        # For buffers, identify which ones should always be included
        for name, buffer in self.ema_buffers.items():
            buffer_name = name.split('.')[-1]  # Get the base name
            # Always include critical buffers regardless of only_save_diff
            if not self.only_save_diff or buffer_name in self._ALWAYS_INCLUDE_BUFFERS:
                state_dict[name] = buffer.data
        
        # Add internal state
        state_dict["initted"] = self.initted
        state_dict["step"] = self.step
        
        return state_dict

    def load_state_dict(self, state_dict):
        """Load state dict with EMA parameters."""
        for name, param in state_dict.items():
            if name == "initted":
                self.initted.data.copy_(param)
            elif name == "step":
                self.step.data.copy_(param)
            elif name in self.ema_params:
                self.ema_params[name].data.copy_(param)
            elif name in self.ema_buffers:
                self.ema_buffers[name].data.copy_(param)
