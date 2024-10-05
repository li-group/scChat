"""Legacy API wrapper.

>>> from legacy_api_wrap import legacy_api
>>> @legacy_api('d', 'c')
... def fn(a, b=None, *, c=2, d=1, e=3):
...     return c, d, e
>>> fn(12, 13, 14) == (2, 14, 3)
True
"""

from __future__ import annotations

import sys
from functools import wraps
from inspect import Parameter, signature
from typing import TYPE_CHECKING
from warnings import warn

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import ParamSpec, TypeVar

    P = ParamSpec("P")
    R = TypeVar("R")

INF = float("inf")
POS_TYPES = {Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD}


# The actual returned Callable of course accepts more positional parameters,
# but we want the type to lie so end users don’t rely on the deprecated API.
def legacy_api(
    *old_positionals: str,
    category: type[Warning] = DeprecationWarning,
    stacklevel: int = 2,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Legacy API wrapper.

    You want to change the API of a function:

    >>> def fn(a, b=None, d=1, c=2, e=3):
    ...     return c, d, e

    Add a the decorator, specifying the parameter names that the old function had
    after the new function’s ``*``.
    Feel free to reorder the newly keyword-only parameters, and/or to add more.

    >>> @legacy_api('d', 'c')
    ... def fn(a, b=None, *, c=2, d=1, e=3):
    ...     return c, d, e

    And the function can be called using one of both signatures, raising a warning.

    >>> fn(12, 13, 14) == (2, 14, 3)
    True

    Parameters
    ----------
    old_positionals
        The positional parameter names that the old function had after the new function’s ``*``.
    category
        The warning class to use for the deprecation.
        Typically, you want to use ``DeprecationWarning``, ``PendingDeprecationWarning``,
        ``FutureWarning``, or a custom subclass of those.
    stacklevel
        The stacklevel to use for the deprecation warning.
        By default, the first stack frame is the call site of the wrapped function.
    """

    def wrapper(fn: Callable[P, R]) -> Callable[P, R]:
        sig = signature(fn)
        par_types = [p.kind for p in sig.parameters.values()]
        has_var = Parameter.VAR_POSITIONAL in par_types
        n_required = sum(1 for p in sig.parameters.values() if p.default is Parameter.empty)
        n_positional = sys.maxsize if has_var else sum(1 for p in par_types if p in POS_TYPES)

        @wraps(fn)
        def fn_compatible(*args_all: P.args, **kw: P.kwargs) -> R:
            if len(args_all) <= n_positional:
                return fn(*args_all, **kw)

            args_pos: P.args
            args_pos, args_rest = args_all[:n_positional], args_all[n_positional:]

            if len(args_rest) > len(old_positionals):
                n_max = n_positional + len(old_positionals)
                msg = (
                    f"{fn.__name__}() takes from {n_required} to {n_max} parameters, "
                    f"but {len(args_pos) + len(args_rest)} were given."
                )
                raise TypeError(msg)
            warn(
                f"The specified parameters {old_positionals[:len(args_rest)]!r} are "
                "no longer positional. "
                f"Please specify them like `{old_positionals[0]}={args_rest[0]!r}`",
                category=category,
                stacklevel=stacklevel,
            )
            kw_new: P.kwargs = {**kw, **dict(zip(old_positionals, args_rest))}

            return fn(*args_pos, **kw_new)

        return fn_compatible

    return wrapper
