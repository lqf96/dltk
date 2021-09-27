from __future__ import annotations

from typing import TYPE_CHECKING
from collections.abc import Sequence, Mapping

import math

import torch as th

from dltk.types import T
from .dataset import SeqDataset
from .iter import DataIterator, BatchIterator, iter_serial

if TYPE_CHECKING:
    from typing import Optional

    from dltk.types import K, Iterable
    from .dataset import Dataset
    from .iter import IterStrategy

__all__ = [
    "DataLoader"
]

class DataLoader(Iterable["list[T]"]):
    def __init__(self, dataset: Dataset[K, T], batch_size: int, shuffle: bool = False, 
        keys: Optional[Iterable[K]] = None, drop_last: bool = False,
        gen: th.Generator = th.default_generator, iter_strategy: IterStrategy = iter_serial):
        if keys is None:
            # Make a list of keys if dataset is shuffled
            if isinstance(dataset, Mapping):
                self._dataset_keys = list(dataset) if shuffle else []
            # Keys cannot be inferred for non-collection mapping dataset
            elif not isinstance(dataset, (SeqDataset, Sequence)):
                raise ValueError("keys must be provided for non-collection mapping dataset")

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.keys = keys
        self.drop_last = drop_last
        self.gen = gen
        self.iter_strategy = iter_strategy

    def __len__(self) -> int:
        return math.ceil(len(self.dataset)/self.batch_size)

    def __iter__(self) -> BatchIterator[T]:
        dataset = self.dataset
        shuffle = self.shuffle
        keys = self.keys
        gen = self.gen
        iter_strategy = self.iter_strategy

        if keys is None:
            # Sequential dataset
            if isinstance(dataset, (SeqDataset, Sequence)):
                it = DataIterator.seq_shuffled(dataset, gen, iter_strategy) \
                    if shuffle \
                    else DataIterator.seq_iter(dataset, iter_strategy)
            # Mapping collection dataset
            elif isinstance(dataset, Mapping):
                if shuffle:
                    keys = DataIterator.seq_shuffled(self._dataset_keys, gen)
                    it = DataIterator(dataset, keys, iter_strategy)
                else:
                    it = DataIterator.map_iter(dataset, iter_strategy)
        # Other mapping datasets
        else:
            it = DataIterator(dataset, keys, iter_strategy)
        
        return BatchIterator(it, self.batch_size, self.drop_last)
