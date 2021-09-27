from typing import TYPE_CHECKING, TypeVar

import sys

K = TypeVar("K")
T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)

__all__ = (
    "Iterable",
    "Sequence",
    "T_co"
)

if TYPE_CHECKING:
    from typing import Any
    from collections.abc import Callable

    Args = tuple[Any, ...]
    # Keyword arguments
    Kwargs = dict[str, Any]
    # Factory function
    Factory = Callable[..., T_co]

# Abstract base classes from Python 3.9+ supports generics
if sys.version_info>=(3, 9):
    from collections.abc import Iterable, Iterator, Mapping
else:
    from typing import Generic
    from collections import abc

    class Iterable(abc.Iterable, Generic[T_co]):
        pass

    class Iterator(abc.Iterator, Generic[T_co]):
        pass

    class Mapping(abc.Mapping, Generic[T_co]):
        pass
