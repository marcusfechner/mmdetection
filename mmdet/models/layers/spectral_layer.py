from typing import Dict, Optional

import torch
import torch.nn as nn
from mmcv.cnn import build_conv_layer as base_build_conv_layer
from torch.nn.functional import normalize
from torch.nn.utils.spectral_norm import SpectralNorm, SpectralNormLoadStateDictPreHook, SpectralNormStateDictHook


class SoftSpectralNorm(SpectralNorm):
    """Spectral normalization as soft constraint."""
    def __init__(
        self,
        name: str = 'weight',
        n_power_iterations: int = 1,
        dim: int = 0,
        coeff: float = 3.0,
        eps: float = 1e-12
        ) -> None:
        self.coeff = coeff
        super().__init__(name, n_power_iterations, dim, eps)
    
    def compute_weight(self, module: nn.Module, do_power_iteration: bool) -> torch.Tensor:
        weight = getattr(module, self.name + '_orig')
        u = getattr(module, self.name + '_u')
        v = getattr(module, self.name + '_v')
        weight_mat = self.reshape_weight_to_matrix(weight)

        if do_power_iteration:
            with torch.no_grad():
                for _ in range(self.n_power_iterations):
                    # Spectral norm of weight equals to `u^T W v`, where `u` and `v`
                    # are the first left and right singular vectors.
                    # This power iteration produces approximations of `u` and `v`.
                    v = normalize(torch.mv(weight_mat.t(), u), dim=0, eps=self.eps, out=v)
                    u = normalize(torch.mv(weight_mat, v), dim=0, eps=self.eps, out=u)
                if self.n_power_iterations > 0:
                    # See above on why we need to clone
                    u = u.clone(memory_format=torch.contiguous_format)
                    v = v.clone(memory_format=torch.contiguous_format)

        sigma = torch.dot(u, torch.mv(weight_mat, v))
        factor = torch.max(torch.ones(1, device=weight.device), sigma / self.coeff)
        weight = weight / factor
        return weight
    
    @staticmethod
    def apply(module: nn.Module, name: str, n_power_iterations: int, dim: int, coeff: float, eps: float) -> 'SoftSpectralNorm':
        for hook in module._forward_pre_hooks.values():
            if isinstance(hook, SoftSpectralNorm) and hook.name == name:
                raise RuntimeError(f"Cannot register two spectral_norm hooks on the same parameter {name}")

        fn = SoftSpectralNorm(name, n_power_iterations, dim, coeff, eps)
        weight = module._parameters[name]
        if weight is None:
            raise ValueError(f'`SpectralNorm` cannot be applied as parameter `{name}` is None')
        if isinstance(weight, torch.nn.parameter.UninitializedParameter):
            raise ValueError(
                'The module passed to `SpectralNorm` can\'t have uninitialized parameters. '
                'Make sure to run the dummy forward before applying spectral normalization')

        with torch.no_grad():
            weight_mat = fn.reshape_weight_to_matrix(weight)

            h, w = weight_mat.size()
            # randomly initialize `u` and `v`
            u = normalize(weight.new_empty(h).normal_(0, 1), dim=0, eps=fn.eps)
            v = normalize(weight.new_empty(w).normal_(0, 1), dim=0, eps=fn.eps)

        delattr(module, fn.name)
        module.register_parameter(fn.name + "_orig", weight)
        # We still need to assign weight back as fn.name because all sorts of
        # things may assume that it exists, e.g., when initializing weights.
        # However, we can't directly assign as it could be an nn.Parameter and
        # gets added as a parameter. Instead, we register weight.data as a plain
        # attribute.
        setattr(module, fn.name, weight.data)
        module.register_buffer(fn.name + "_u", u)
        module.register_buffer(fn.name + "_v", v)

        module.register_forward_pre_hook(fn)
        module._register_state_dict_hook(SpectralNormStateDictHook(fn))
        module._register_load_state_dict_pre_hook(SpectralNormLoadStateDictPreHook(fn))
        return fn

def soft_spectral_norm(module: nn.Module,
                  name: str = 'weight',
                  n_power_iterations: int = 1,
                  coeff: float = 3.0,
                  eps: float = 1e-12,
                  dim: Optional[int] = None) -> nn.Module:
    if dim is None:
        if isinstance(module, (torch.nn.ConvTranspose1d,
                               torch.nn.ConvTranspose2d,
                               torch.nn.ConvTranspose3d)):
            dim = 1
        else:
            dim = 0
    SoftSpectralNorm.apply(module, name, n_power_iterations, dim, coeff, eps)
    return module

def wrap_conv(module: nn.Module, spectral_norm: bool = False) -> nn.Module:
    """Wrap convolution layer with spectral normalization.
    
    Info: Convs have no bias
    
    Args:
        module (nn.Module): Convolution layer.
        spectral_norm (bool): Whether to use spectral normalization.
    
    Returns:
        nn.Module: Wrapped convolution layer.
    """
    if spectral_norm:
        return soft_spectral_norm(module)
    return module

def build_conv_layer(cfg: Optional[Dict], *args, **kwargs) -> nn.Module:
    """Build convolution layer with soft spectral normalization.

    Args:
        cfg (None or dict): The conv layer config, which should contain:
            - type (str): Layer type.
            - layer args: Args needed to instantiate an conv layer.
        args (argument list): Arguments passed to the `__init__`
            method of the corresponding conv layer.
        kwargs (keyword arguments): Keyword arguments passed to the `__init__`
            method of the corresponding conv layer.

    Returns:
        nn.Module: Created conv layer.
    """
    if cfg is None:
        cfg = dict(type='Conv2d')
    spectral_norm = cfg.pop('spectral_norm', False)
    conv_layer = base_build_conv_layer(cfg, *args, **kwargs)
    return wrap_conv(conv_layer, spectral_norm) if spectral_norm else conv_layer