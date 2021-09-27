from __future__ import annotations

from typing import TYPE_CHECKING
import logging
from logging import Formatter, StreamHandler

if TYPE_CHECKING:
    from typing import Optional

    from logging import Handler

__all__ = (
    "DEFAULT_LOG_FORMAT",
    "setup_log_handler"
)

# Default logging format
DEFAULT_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"

def setup_log_handler(log_handler: Optional[Handler] = None, format: str = DEFAULT_LOG_FORMAT,
    level: int = logging.INFO) -> Handler:
    # Create a stream log handler by 
    log_handler = log_handler or StreamHandler()

    # Set log format
    log_handler.setFormatter(Formatter(DEFAULT_LOG_FORMAT))
    # Set log level
    log_handler.setLevel(level)

    return log_handler
