from __future__ import annotations

from typing import TYPE_CHECKING, Generic
from dataclasses import dataclass, field

import math
from torch.distributed import rpc
from torch.futures import wait_all

from dltk.distributed import DistGroup
from dltk.engine import Event, LocalEngine, SLContext
from dltk.engine.events import AbstractEventTarget
from dltk.engine.sl import SLEngineBase

if TYPE_CHECKING:
    from typing import Any, Optional, Union
    from collections.abc import Callable

    from torch.nn import Module, Parameter
    from torch.optim import Optimizer

    from dltk.types import T, Args, Kwargs, Iterable
    from dltk.engine.base import Context, LRSchedArgs
    from dltk.engine.events import Handler, Filter
    from dltk.engine.sl import IterFunc

__all__ = [
    "DistEngine",
    "DistEventTarget",
    "DistSLEngine",
    "RunOnNodes",
    "WorkerEngine"
]

class WorkerEngine(LocalEngine, Generic[T]):
    ctx: Context

    def __init__(self, iter_func: IterFunc[T], model: Module, group: DistGroup, **kwargs: Any):
        super().__init__(model, **kwargs)

        self.iter_func = iter_func
        self.group = group

    def _set_ctx(self, ctx: Context):
        self.ctx = ctx
    
    def _unset_ctx(self):
        del self.ctx

    def _run_iter(self, epoch_count: int, iter_count: int, shard: list[T]):
        ctx = self.ctx
        # Synchronize epoch and iteration count
        ctx.epoch_count = epoch_count
        ctx.iter_count = iter_count

        # Run iteration function
        self.iter_func(self, shard)

    def average_grads(self, weight: float = 1.):
        # Unscale gradients before averaging
        self.unscale_grads()

        # Collect gradients
        grads = [p.grad for p in self.model.parameters() if p.grad is not None]
        # Perform weighted averaging on gradients
        self.group.weighted_avg(grads, weight)

class DistEventTarget(AbstractEventTarget):
    _workers: list

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

        self._handler_counts: dict[Event, int] = {}

    def on(self, event: Event, handler: Handler, *, args: Args = (), kwargs: Kwargs = {},
        filters: Iterable[Filter] = ()) -> int:
        handler_counts = self._handler_counts

        # Increase handler count
        handler_counts.setdefault(event, 0)
        handler_counts[event] += 1
        # Add event handler on remote nodes
        wait_all(
            worker.rpc_async().on(event, handler, args, kwargs, filters) \
            for worker in self._workers
        )

    def off(self, event: Event, handler_id: int = 0) -> bool:
        handler_counts = self._handler_counts

        # Decrease handler count
        if handler_id==0:
            del handler_counts[event]
        else:
            handler_counts[event] -= 1
        # Remove event handler on remote nodes
        wait_all(
            worker.rpc_async().off(event, handler_id) for worker in self._workers
        )

    def emit(self, event: Event, *args: Any, **kwargs: Any):
        # Do not emit event when no handler is added
        if self._handler_counts.get(event, 0)==0:
            return
        # Emit event on remote nodes
        wait_all(
            worker.rpc_async().emit(event, *args, **kwargs) for worker in self._workers
        )

class DistSLEngine(SLEngineBase, DistEventTarget):
    def __init__(self, iter_func: IterFunc, model: Module, world_size: int, *, **kwargs: Any):
        super().__init__()

        self.iter_func = iter_func
        self.world_size = world_size

        # Worker engines on remote nodes
        self._workers = [
            rpc.remote(i, WorkerEngine, args=(iter_func, model, i), kwargs=kwargs) \
            for i in range(world_size)
        ]
        # Training flag
        self._training = model.training

    def _set_ctx(self, ctx: SLContext):
        super()._set_ctx(ctx)

        # Set context on remote nodes
        wait_all(worker.rpc_async()._set_ctx(ctx) for worker in self._workers)
    
    def _unset_ctx(self):
        super()._unset_ctx()

        # Unset context on remote nodes
        wait_all(worker.rpc_async()._unset_ctx() for worker in self._workers)

    def _run_iter(self, batch: list[T]):
        world_size = self.world_size
        ctx = self.ctx

        # Split batch of samples into shards for all nodes
        shard_size = math.ceil(len(batch)/world_size)
        iter_futures = []
        # Run iteration with shard for each node
        for i, worker in enumerate(self._workers):
            shard = batch[i*shard_size:(i+1)*shard_size]
            iter_futures.append(worker.rpc_async()._run_iter(
                ctx.epoch_count, ctx.iter_count, shard
            ))
        
        # Wait for iteration to complete on remote nodes
        wait_all(iter_futures)

    @property
    def training(self) -> bool:
        return self._training
    
    def train(self, training: bool = True):
        self._training = training
        # Set training flag on remote nodes
        wait_all(worker.rpc_async().train(training) for worker in self._workers)

    def create_optimizer(self, name: str, optim_factory: Callable[..., Optimizer],
        params: Union[Iterable[Parameter], Iterable[Kwargs]], optim_kwargs: Kwargs = {},
        lr_sched_args: Iterable[LRSchedArgs] = ()):
        # Create optimizers on remote nodes
        wait_all(worker.rpc_async().create_optimizer(
            name, optim_factory, params, optim_kwargs, lr_sched_args
        ) for worker in self._workers)

@dataclass
class RunOnNodes:
    ranks: list[int] = field(default_factory=lambda: [0])

    def __call__(self, engine: WorkerEngine, *args: Any, **kwargs: Any) -> bool:
        return engine.group.rank in self.ranks
