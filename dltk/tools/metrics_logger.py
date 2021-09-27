from __future__ import annotations
from dltk.distributed.engine import RunOnNodes

from typing import TYPE_CHECKING

from torchmetrics import MetricCollection

from dltk.engine import Event, EventHandler, Every
from dltk.distributed import WorkerEngine

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter
    from torchmetrics import Metric

    from dltk.engine import LocalEngine
    from dltk.engine.events import Filter

__all__ = [
    "TensorBoardLogger"
]

class TensorBoardLogger(EventHandler):
    def __init__(self, metrics: MetricCollection, writer: SummaryWriter,
        log_event: Event = Event.ITER_ENDED, interval: int = 1, ranks: list[int] = [0]):
        self.metrics = metrics
        self.writer = writer
        self.log_event = log_event
        self.interval = interval
        self.ranks = ranks

    def _log_metrics(self, engine: LocalEngine):
        writer = self.writer
        ctx = engine.ctx

        # Get current engine mode
        mode = ctx.mode.value
        # Compute and log metrics
        metric: Metric
        for name, metric in self.metrics.items():
            writer.add_scalar(f"{mode}/{name}", metric.compute(), ctx.iter_count)

    def attach(self, engine: LocalEngine):
        super().attach(engine)

        interval = self.interval
        log_event = self.log_event

        filters: list[Filter] = []
        # Log metrics for selected nodes only under distributed settings
        if isinstance(engine, WorkerEngine):
            filters.append(RunOnNodes(self.ranks))
        # Logging interval
        if interval>1:
            filters.append(Every(interval, log_event))

        self._on(self.log_event, self._log_metrics, filters=filters)
