from __future__ import annotations

from typing_extensions import Protocol, runtime_checkable
from dltk.types import K, T

from .iter import DataIterator

__all__ = [
    "Dataset",
    "SeqDataset"
]

@runtime_checkable
class Dataset(Protocol[K, T]):
    def __len__(self) -> int:
        raise NotImplementedError
    
    def __getitem__(self, key: K) -> T:
        raise NotImplementedError

class SeqDataset(Dataset[int, T]):
    def __iter__(self) -> DataIterator[T]:
        return DataIterator.seq_iter(self)
