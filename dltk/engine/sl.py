from __future__ import annotations

from typing import TYPE_CHECKING, Generic
from dltk.types import Iterable, T

from tqdm import tqdm

from .base import LocalEngine, Context, ExecMode
from .events import Event

if TYPE_CHECKING:
    from typing import Any
    from collections.abc import Callable

    IterFunc = Callable[[LocalEngine, list[T]], None]

__all__ = [
    "SLEngine",
    "SLContext"
]

class SLContext(Context):
    def __init__(self, n_epochs: int, batch_size: int, **kwargs: Any):
        super().__init__(**kwargs)

        self.n_epochs = n_epochs
        self.batch_size = batch_size
        
        # Number of iteration assigned when switching to mode
        self.n_iters = 0

def _infer_n_iters(loader: Iterable[list[T]], n_iters: int, mode: ExecMode) -> int:
    # Loader has number of iterations
    try:
        n_iters = len(loader)
    # Fall back to explicitly provided number of iterations for unsized dataset
    except (TypeError, NotImplementedError):
        if n_iters<=0:
            raise ValueError(f"number of iterations required for unsized {mode.value} dataset")
    
    return n_iters

class SLEngineBase(LocalEngine, Generic[T]):
    ctx: SLContext

    def _run_iter(self, batch: list[T]):
        raise NotImplementedError

    def _set_ctx(self, ctx: SLContext):
        self.ctx = ctx
    
    def _unset_ctx(self):
        del self.ctx

    def _run(self, ctx: SLContext, modes: dict[str, tuple[Iterable[list[T]], int]]):
        # Set supervised learning context
        self._set_ctx(ctx)

        while ctx.epoch_count<ctx.n_epochs:
            # Update epoch count
            ctx.epoch_count += 1
            # Emit epoch started event
            self.emit(Event.EPOCH_STARTED)

            for mode, (dataset, n_iters) in modes.items():
                #  Update mode and iteration count
                ctx.mode = mode
                ctx.iter_count = 0
                # Set engine to training mode
                if mode=="train":
                    self.train()
                # Emit mode started event
                self.emit(Event.MODE_STARTED)

                for batch in tqdm(dataset, desc=f"Epoch {ctx.epoch_count}: {mode}", total=n_iters):
                    # Update iteration count
                    ctx.iter_count += 1
                    # Emit iteration started event
                    self.emit(Event.ITER_STARTED)
                    # Run an iteration
                    self._run_iter(batch)
                    # Emit batch ended event
                    self.emit(Event.ITER_ENDED)

                # Emit mode ended event
                self.emit(Event.MODE_ENDED)
                # Restore engine to evaluation mode
                if mode=="train":
                    self.eval()

            # Emit epoch ended event
            self.emit(Event.EPOCH_ENDED)
        
        # Unset supervised learning context
        self._unset_ctx()

    def fit(self, train_loader: Iterable[list[T]], dev_loader: Iterable[list[T]], n_epochs: int,
        batch_size: int, n_train_iters: int = 0, n_dev_iters: int = 0, **kwargs):
        # Infer number of training and validation iterations
        n_train_iters = _infer_n_iters(train_loader, n_train_iters, ExecMode.TRAIN)
        n_dev_iters = _infer_n_iters(dev_loader, n_dev_iters, ExecMode.DEV)

        # Create and run with supervised learning context
        self._run(SLContext(n_epochs, batch_size, **kwargs), {
            ExecMode.TRAIN: (train_loader, n_train_iters),
            ExecMode.DEV: (dev_loader, n_dev_iters)
        })

    def predict(self, loader: Iterable[list[T]], batch_size: int, n_iters: int = 0, **kwargs: Any):
        # Infer number of prediction iterations
        n_iters = _infer_n_iters(loader, n_iters, ExecMode.EVAL)

        # Create and run with supervised learning context
        self._run(SLContext(1, batch_size, **kwargs), {ExecMode.EVAL: (loader, n_iters)})
