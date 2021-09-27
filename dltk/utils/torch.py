from __future__ import annotations

import torch as th
from torch.cuda.amp import GradScaler

__all__ = (
    "autocast_cls",
    "grad_scaler"
)

def autocast_cls(device: str):
    # CUDA
    if device=="cuda":
        return th.cuda.amp.autocast
    # CPU
    elif device=="cpu":
        return th.cpu.amp.autocast
    # XLA
    elif device=="xla":
        from torch_xla.amp import autocast as autocast_xla
        return autocast_xla
    # Unknown device
    else:
        raise ValueError(f"device type '{device}' does not support AMP")

def grad_scaler(device: str, enabled: bool = True) -> GradScaler:
    # XLA
    if device=="xla":
        from torch_xla.amp import GradScaler as GradScalerXLA
        return GradScalerXLA(enabled=enabled)
    # Other devices
    else:
        return GradScaler(enabled=enabled)
