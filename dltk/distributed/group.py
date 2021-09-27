from __future__ import annotations

from typing import TYPE_CHECKING
from typing_extensions import Protocol
from abc import abstractmethod

from functools import cached_property

import torch as th
from torch import distributed as dist
from torch.distributed import rpc
from torch.futures import wait_all

try:
    from torch_xla.core import xla_model as xm
except ImportError:
    pass

if TYPE_CHECKING:
    from torch.distributed import ReduceOp

__all__ = [
    "DistGroup",
    "DistGroupTorch",
    "DistGroupXLA"
]

class DistGroup(Protocol):
    @property
    @abstractmethod
    def size(self) -> int:
        raise NotImplementedError
    
    @property
    @abstractmethod
    def rank(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def all_reduce(self, tensors: list[th.Tensor], op: ReduceOp = ReduceOp.SUM):
        raise NotImplementedError
    
    def weighted_avg(self, tensors: list[th.Tensor], weight: float = 1.):
        # Current node does not participate in averaging
        if weight==0:
            for tensor in tensors:
                tensor.zero_()
        
        # Make a weight tensor
        weight_tensor = tensors[-1].new(weight)
        tensors.append(weight_tensor)
        # All-reduce sum all tensors
        self.all_reduce(tensors, op=ReduceOp.SUM)
        
        # Pop weight tensor
        tensors.pop()
        # Compute average
        for tensor in tensors:
            tensor /= weight_tensor

class DistGroupTorch(DistGroup):
    _torch_groups = {(): None}

    def __init__(self, ranks: list[int] = []):
        self._ranks = tuple(ranks)
    
    @classmethod
    def _new_group(cls, ranks: tuple[int, ...]):
        # Create and store underlying distributed group
        cls._torch_groups[ranks] = dist.new_group(ranks)

    @cached_property
    def _torch_group(self):
        ranks = self._ranks
        torch_groups = self._torch_groups

        # Global distributed group
        if ranks==():
            return None
        # Existing distributed group
        group = torch_groups.get(ranks)
        if group:
            return group
        
        # Create distributed group on all nodes using RPC
        wait_all(
            rpc.rpc_async(i, self._new_group, (ranks,)) \
            for i in range(dist.get_world_size())
        )
        # Newly created group
        return torch_groups[ranks]

    @property
    def size(self) -> int:
        return dist.get_world_size(self._torch_group(self._ranks))
    
    @property
    def rank(self) -> int:
        return dist.get_rank(self._torch_group(self._ranks))

    def all_reduce(self, tensors: list[th.Tensor], op: ReduceOp = ReduceOp.SUM):
        group = self._torch_group(self._ranks)

        # Perform all-reduce asynchronously for all tensors
        futures = [dist.all_reduce(tensor, op, group, async_op=True) for tensor in tensors]
        # Wait for all operations to complete
        for future in futures:
            future.wait()

_XLA_REDUCE_OP = {
    ReduceOp.SUM: xm.REDUCE_SUM,
    ReduceOp.PRODUCT: xm.REDUCE_MUL,
    ReduceOp.MIN: xm.REDUCE_MIN,
    ReduceOp.MAX: xm.REDUCE_MAX
}

class DistGroupXLA(DistGroup):
    def __init__(self, ranks: list[int] = []):
        self._ranks = ranks

    @property
    def size(self) -> int:
        ranks = self._ranks
        # Empty ranks imply global group
        return len(ranks) if ranks else xm.xrt_world_size()

    @property
    def rank(self) -> int:
        ranks = self._ranks

        global_rank = xm.get_ordinal()
        # Empty ranks imply global group
        return ranks.index(global_rank) if ranks else global_rank

    def all_reduce(self, tensors: list[th.Tensor], op: ReduceOp = ReduceOp.SUM):
        ranks = self._ranks

        xla_reduce_op = _XLA_REDUCE_OP.get(op)
        # Unsupported operation
        if xla_reduce_op is None:
            raise RuntimeError(f"unsupported reduction operation for XLA: {xla_reduce_op}")
        
        groups = [ranks] if ranks else None
        # Perform all-reduce for all tensors
        xm.all_reduce(xla_reduce_op, tensors, groups)
