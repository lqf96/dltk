from __future__ import annotations

from typing import TYPE_CHECKING

import time
from logging import getLogger
from logging.handlers import QueueHandler, QueueListener
from argparse import ArgumentParser

import torch as th
from torch import cuda, distributed as dist
from torch.distributed import rpc
from torch import multiprocessing as mp

from dltk.utils import setup_log_handler
from .group import DistGroup, DistGroupTorch, DistGroupXLA

try:
    from torch_xla.distributed import xla_multiprocessing as mp_xla
except ImportError:
    pass

if TYPE_CHECKING:
    from collections.abc import Callable

    from dltk.types import Iterable

    NodeCallback = Callable[[th.device, DistGroup], None]

__all__ = [
    "DEV_TYPES",
    "DEFAULT_DEV_TYPE",
    "DEFAULT_ADDR",
    "DEFAULT_PORT",
    "DEFAULT_DIST_BACKEND",
    "launch_nodes",
    "add_dist_cli_args"
]

_logger = getLogger(__name__)

def _launch_node(local_rank: int, world_size: int, rank_start: int, device_type: str,
    device_indices: Iterable[int], address: str, port: int, dist_backend: str, log_queue: mp.Queue,
    callback: NodeCallback):
    # Set up logging for node process
    _logger.addHandler(setup_log_handler(QueueHandler(log_queue)))

    # Rank and device index
    rank = rank_start+local_rank
    device_idx = device_indices[local_rank] if device_indices else local_rank
    # Set CUDA device to avoid NCCL error
    if device_type=="cuda":
        cuda.set_device(device_idx)

    _logger.info(
        f"Starting node {rank} (Local node {local_rank}) on device '{device_type}:{device_idx}' ..."
    )
    # Initialize distributed group
    if device_type!="xla":
        dist.init_process_group(
            dist_backend, init_method=f"tcp://{address}:{port}", world_size=world_size, rank=rank
        )
    # Initialize RPC group
    rpc.init_rpc(
        name=f"node{rank}",
        world_size=world_size,
        rank=rank,
        rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
            init_method=f"tcp://{address}:{port+1}"
        )
    )
    # Node started
    _logger.info(f"Node {rank} started.")

    # Device and distributed group
    device = th.device(device_type, device_idx)
    dist_group = DistGroupXLA() if device_type=="xla" else DistGroupTorch()
    # Invoke callback
    try:
        callback(device, dist_group)
    # Node stopped by parent process
    except KeyboardInterrupt:
        pass
    # Log error
    except Exception:
        _logger.exception(f"Error occurred on node {rank}:")

    _logger.info(f"Stopping node {rank} ...")
    # Destroy RPC group
    rpc.shutdown()
    # Destroy distributed group
    dist.destroy_process_group()
    # Node stopped
    _logger.info(f"Node {rank} stopped.")

def _idle(interval: float = 10.):
    try:
        while True:
            time.sleep(interval)
    except KeyboardInterrupt:
        pass

def _parse_device_indices(raw_indices: str) -> list[int]:
    return [int(raw_idx) for raw_idx in raw_indices.split(",")]

# Available evice types
DEV_TYPES = ("cpu", "cuda", "xla")
# Defaults
DEFAULT_DEV_TYPE = "cuda"
DEFAULT_ADDR = "0.0.0.0"
DEFAULT_PORT = 12345
DEFAULT_DIST_BACKEND = "nccl"

def launch_nodes(world_size: int, rank_start: int = 0, device_type: str = DEFAULT_DEV_TYPE,
    device_indices: Iterable[int] = (), address: str = DEFAULT_ADDR, port: int = DEFAULT_PORT,
    dist_backend: str = DEFAULT_DIST_BACKEND, callback: NodeCallback = _idle):
    # Check device type
    if device_type not in DEV_TYPES:
        raise RuntimeError(f"unknown device type: '{device_type}'")
    # Infer device indices for CPU and CUDA
    if not device_indices:
        if device_type=="cpu":
            device_indices = [0]
        elif device_type=="cuda":
            device_indices = range(cuda.device_count())
    
    _logger.info(f"Starting {len(device_indices)} nodes ({world_size} total) ...")
    _logger.info(f"Connecting to primary node at {address}:{{{port}, {port+1}}} ...")

    # Create and start log forwarder
    log_queue = mp.Queue()
    log_listener = QueueListener(log_queue)
    log_listener.start()
    # Start node processes
    spawn_args = (
        world_size, rank_start, device_type, device_indices, address, port,
        dist_backend, log_queue, callback
    )
    if device_type=="xla":
        mp_xla.spawn(_launch_node, args=spawn_args)
    else:
        mp.spawn(_launch_node, args=spawn_args, nprocs=len(device_indices))

def add_dist_cli_args(parser: ArgumentParser) -> ArgumentParser:
    # Distributed CLI arguments
    parser.add_argument(
        "-w", "--world-size", type=int, help="Number of workers in the distributed group."
    )
    parser.add_argument(
        "-r", "--rank-start", type=int, help="Starting rank of all workers on this machine."
    )
    parser.add_argument(
        "-i", "--device-indices", type=_parse_device_indices, default=(),
        help="Indices of all devices to be used for execution, separated by commas."
    )
    parser.add_argument(
        "-d", "--device-type", choices=DEV_TYPES, default=DEFAULT_DEV_TYPE,
        help="Type of device to be used for execution."
    )
    parser.add_argument(
        "-a", "--address", default=DEFAULT_ADDR, help="IP address of the initial worker."
    )
    parser.add_argument(
        "-p", "--port", type=int, default=DEFAULT_PORT, help="Port of the initial worker."
    )
    parser.add_argument(
        "-b", "--backend", choices=("gloo", "nccl", "mpi"), default=DEFAULT_DIST_BACKEND,
        help="PyTorch distributed backend to use."
    )

    return parser

def dist_main():
    parser = add_dist_cli_args(ArgumentParser())
    # Parse distributed CLI arguments
    args = parser.parse_args()

    # Launch node
    launch_nodes(**args.__dict__)
