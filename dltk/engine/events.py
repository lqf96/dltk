from __future__ import annotations

from typing import TYPE_CHECKING
from typing_extensions import Protocol
from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum

if TYPE_CHECKING:
    from typing import Any, Optional
    from collections.abc import Callable

    from dltk.types import Args, Kwargs, Iterable

    Handler = Callable[..., Any]
    Filter = Callable[..., bool]

__all__ = [
    "AbstractEventTarget",
    "Event",
    "EventHandler",
    "EventTarget"
]

@dataclass
class _HandlerInfo:
    handler: Handler
    args: Args
    kwargs: Kwargs
    filters: tuple[Filter, ...]

class Event(Enum):
    EPOCH_STARTED = "epoch_started"
    EPOCH_ENDED = "epoch_ended"
    MODE_STARTED = "mode_started"
    MODE_ENDED = "mode_ended"
    ITER_STARTED = "iter_started"
    ITER_ENDED = "iter_ended"

class AbstractEventTarget(Protocol):
    @abstractmethod
    def on(self, event: Event, handler: Handler, *, args: Args = (), kwargs: Kwargs = {},
        filters: Iterable[Filter] = ()) -> int:
        raise NotImplementedError
    
    @abstractmethod
    def off(self, event: Event, handler_id: int = 0) -> bool:
        raise NotImplementedError
    
    @abstractmethod
    def emit(self, event: Event, *args: Any, **kwargs: Any):
        raise NotImplementedError

class EventTarget(AbstractEventTarget):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

        self._event_handlers: dict[Event, dict[int, _HandlerInfo]] = {}

    def on(self, event: Event, handler: Handler, *, args: Args = (), kwargs: Kwargs = {},
        filters: Iterable[Filter] = ()) -> int:
        # Create handler infomation and use its address as handler ID
        handler_info = _HandlerInfo(handler, tuple(args), kwargs, tuple(filters))
        handler_id = id(handler_info)
        # Add handler information to store
        event_handlers = self._event_handlers.setdefault(event, {})
        event_handlers[handler_id] = handler_info

        return handler_id

    def off(self, event: Event, handler_id: int = 0) -> bool:
        event_handlers = self._event_handlers

        # Remove all handlers for event
        if handler_id==0:
            handlers = event_handlers.pop(event, None)
            return bool(handlers)
        # Remove handler by event and ID
        else:
            info = event_handlers.get(event, {}).pop(handler_id, None)
            return bool(info)

    def emit(self, event: Event, *args: Any, **kwargs: Any):
        handlers = self._event_handlers.get(event)
        # No handlers registered for current event
        if not handlers:
            return

        for info in handlers.values():
            # Call filters to determine if handler should be invoked
            invoke_handler = all(filter(self, *args, event=event, **kwargs) for filter in info.filters)
            if not invoke_handler:
                continue

            # Call event handler
            info.handler(self, *info.args, *args, **info.kwargs, **kwargs)

class EventHandler:
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

        self._target: Optional[EventTarget] = None
        self._handler_ids: list[tuple[Event, int]] = []

    def _on(self, event: Event, handler: Handler, *, args: Args = (), kwargs: Kwargs = {},
        filters: Iterable[Filter] = ()):
        # Register event handler
        handler_id = self._target.on(event, handler, args, kwargs, filters)
        # Save event and handler ID
        self._handler_ids.append((event, handler_id))

    def attach(self, target: EventTarget):
        # Check and set event target
        if self._target is not None:
            raise RuntimeError(f"event handlers already registered for target {target}")
        self._target = target

    def detach(self):
        target = self._target
        handler_ids = self._handler_ids
        # Do nothing if event handlers are not registered
        if target is None:
            return
        
        # Unregister event handlers
        for event, handler_id in handler_ids:
            target.off(event, handler_id)
        # Clear handler IDs
        handler_ids.clear()
