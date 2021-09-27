from __future__ import annotations

from typing import TYPE_CHECKING, Generic
from dltk.types import K, T, Iterable, Iterator, Mapping

import torch as th

if TYPE_CHECKING:
    from collections.abc import Callable

    from .dataset import Dataset, SeqDataset

    IterStrategy = Callable[[Dataset[K, T], Iterator[K]], Iterator[T]]

__all__ = [
    "DataIterator",
    "BatchIterator",
    "iter_serial"
]

def iter_serial(dataset: Dataset[K, T], keys: Iterator[K]) -> Iterator[T]:
    return map(dataset.__getitem__, keys)

class DataIterator(Generic[K, T], Iterator[T]):
    def __init__(self, dataset: Dataset[K, T], keys: Iterable[K],
        iter_strategy: IterStrategy = iter_serial):
        self._iter = iter_strategy(dataset, iter(keys))

    def __next__(self) -> T:
        return next(self._iter)

    @classmethod
    def seq_iter(cls, dataset: SeqDataset[T], iter_strategy: IterStrategy = iter_serial):
        return cls(dataset, range(len(dataset)), iter_strategy)

    @classmethod
    def seq_shuffled(cls, dataset: SeqDataset[T], gen: th.Generator = th.default_generator,
        iter_strategy: IterStrategy = iter_serial):
        # Random permutation of indices (converted to memory view for serializable iterator)
        perm_indices = memoryview(th.randperm(len(dataset), generator=gen).numpy())

        return cls(dataset, perm_indices, iter_strategy)

    @classmethod
    def map_iter(cls, dataset: Mapping[K, T], iter_strategy: IterStrategy = iter_serial):
        return cls(dataset, iter(dataset), iter_strategy)

class BatchIterator(Iterator["list[T]"]):
    def __init__(self, data: Iterable[T], batch_size: int, drop_last: bool = False):
        self.batch_size = batch_size
        self.drop_last = drop_last

        self._iter = iter(data)
    
    def __next__(self) -> list[T]:
        it = self._iter
        batch: list[T] = []

        try:
            # Collect next batch of samples
            for _ in range(self.batch_size):
                batch.append(next(it))
        except StopIteration:
            # Return incomplete batch
            if batch and not self.drop_last:
                return batch
            # End of underlying iterator
            else:
                raise
