from __future__ import annotations

from typing import TYPE_CHECKING
from abc import abstractmethod
from enum import Enum
from dataclasses import dataclass

from types import SimpleNamespace
from contextlib import contextmanager

import torch as th

from dltk.utils import autocast_cls, grad_scaler
from .events import AbstractEventTarget, Event, EventTarget

if TYPE_CHECKING:
    from typing import Any, Union
    from collections.abc import Callable, Iterable, Iterator

    from torch.nn import Module, Parameter
    from torch.nn.utils.clip_grad import clip_grad_norm_, clip_grad_value_
    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import _LRScheduler as LRScheduler

    from dltk.types import Kwargs, Factory

    LRSchedArgs = tuple[Factory[LRScheduler], Kwargs]

__all__ = [
    "AbstractEngine",
    "Context",
    "Every",
    "ExexMode",
    "LocalEngine"
]

class ExecMode(Enum):
    TRAIN = "train"
    DEV = "dev"
    EVAL = "evaluate"

    UNKNOWN = "<unknown>"

class Context(SimpleNamespace):
    _event_to_count = {
        Event.EPOCH_STARTED: "epoch",
        Event.EPOCH_ENDED: "epoch",
        Event.ITER_STARTED: "iter",
        Event.ITER_ENDED: "iter"
    }

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

        self.epoch_count = 0
        self.iter_count = 0
        self.mode = ExecMode.UNKNOWN

    def count_for(self, event: Event) -> int:
        return getattr(self, self._event_to_count[event])

class AbstractEngine(AbstractEventTarget):
    @property
    @abstractmethod
    def training(self) -> bool:
        raise NotImplementedError
    
    @abstractmethod
    def train(self, training: bool = True):
        raise NotImplementedError
    
    def eval(self):
        self.train(False)

class LocalEngine(AbstractEngine, EventTarget):
    ctx: Context

    def __init__(self, model: Module, device: Union[th.device, str], enable_amp: bool = True):
        super().__init__()

        device = th.device(device)
        # Device for current node
        self.device = device
        # Model on given device
        self.model = model.to(device)

        # AMP autocast class
        self._autocast_cls = autocast_cls(device)
        # Gradient scaler
        self._scaler = grad_scaler(device, enabled=enable_amp)
        # Optimizers
        self._optimizers: dict[str, Optimizer] = {}
        # Learning rate schedulers
        self._lr_scheds: dict[str, list[LRScheduler]] = {}

        # Optimizers with gradients unscaled
        self._unscaled_optims: set[str] = set()
    
    def _params_for_optimizers(self, optim_names: Iterable[str]) -> Iterator[Parameter]:
        optimizers = self._optimizers

        # For each selected optimizer ...
        for optim_name in optim_names:
            optimizer = optimizers[optim_name]
            # Yield each parameter in all parameter groups
            for param_group in optimizer.param_groups:
                for param in param_group["params"]:
                    yield param

    @property
    def enable_amp(self) -> bool:
        return self._scaler.is_enabled()

    @property
    def training(self) -> bool:
        return self.model.training

    def train(self, training: bool = True):
        self.model.train(training)

    def state_dict(self) -> dict[str, Any]:
        return {
            "model": self.model.state_dict(),
            "scaler": self._scaler.state_dict(),
            "optimizers": {
                optim_name: optimizer.state_dict() \
                for optim_name, optimizer in self._optimizers.items()
            },
            "lr_scheds": {
                optim_name: [lr_sched.state_dict() for lr_sched in lr_scheds] \
                for optim_name, lr_scheds in self._lr_scheds.items()
            }
        }
    
    def load_state_dict(self, state: dict[str, Any]):
        optimizers = self._optimizers
        lr_scheds = self._lr_scheds

        # Load state for the model
        self.model.load_state_dict(state["model"])
        # Load state for gradient scaler
        self._scaler.load_state_dict(state["scaler"])
        # Load state for optimizers
        for optim_name, optim_state in state["optimizers"]:
            optimizers[optim_name].load_state_dict(optim_state)
        # Load state for learning rate schedulers
        for optim_name, lr_sched_states in state["lr_scheds"]:
            for lr_sched, lr_sched_state in zip(lr_scheds[optim_name], lr_sched_states):
                lr_sched.load_state_dict(lr_sched_state)

    def create_optimizer(self, name: str, optim_factory: Callable[..., Optimizer],
        params: Union[Iterable[Parameter], Iterable[Kwargs]], optim_kwargs: Kwargs = {},
        lr_sched_args: Iterable[LRSchedArgs] = ()):
        # Create optimizer
        self._optimizers[name] = optimizer = optim_factory(params, **optim_kwargs)
        # Create LR schedulers
        self._lr_scheds[name] = [factory(optimizer, **kwargs) for factory, kwargs in lr_sched_args]

    @contextmanager
    def forward_ctx(self) -> Iterator[None]:
        # Enable gradient computation for training and AMP
        with th.set_grad_enabled(self.training), self._autocast_cls(self.enable_amp):
            yield

    def backward(self, *losses: th.Tensor, retain_graph: bool = False, create_graph: bool = False):
        scaler = self._scaler

        for loss in losses:
            scaler.scale(loss).backward(retain_graph=retain_graph, create_graph=create_graph)

    def unscale_grads(self, *optim_names: str):
        scaler = self._scaler
        optimizers = self._optimizers
        unscaled_optims = self._unscaled_optims

        optim_names = optim_names or optimizers.keys()
        # Unscale given (or all) optimizers
        for optim_name in optim_names:
            if optim_name in unscaled_optims:
                continue
            
            scaler.unscale_(optimizers[optim_name])
            # Mark optimized as unscaled
            unscaled_optims.add(optim_name)
        
    def clip_grad_by_norm(self, max_norm: float, *optim_names: str, norm_type: float = 2.):
        self.unscale_grads(*optim_names)

        params = self._params_for_optimizers(optim_names) if optim_names \
            else self.model.parameters()
        # Clip gradients by norm
        clip_grad_norm_(params, max_norm, norm_type)

    def clip_grad_by_value(self, clip_value: float, *optim_names: str):
        self.unscale_grads(*optim_names)

        params = self._params_for_optimizers(optim_names) if optim_names \
            else self.model.parameters()
        # Clip gradients by value
        clip_grad_value_(params, clip_value)

    def step(self, *optim_names: str):
        scaler = self._scaler
        optimizers = self._optimizers
        lr_scheds = self._lr_scheds
        unscaled_optims = self._unscaled_optims

        optim_names = optim_names or optimizers.keys()
        # Step given (or all) optimizers
        for optim_name in optim_names:
            optimizer = optimizers[optim_name]

            scaler.step(optimizer)
            # Step learning rate schedulers
            for lr_sched in lr_scheds[optim_name]:
                lr_sched.step()
            # Clear gradients
            optimizer.zero_grad()

            # Remove optimizer from unscaled optimizers
            if optim_name in unscaled_optims:
                unscaled_optims.remove(optim_name)
        
        # Update gradient scaler
        scaler.update()

@dataclass
class Every:
    interval: int

    def __call__(self, engine: LocalEngine, *args: Any, event: Event, **kwargs: Any):
        return engine.ctx.count_for(event)%self.interval==0
