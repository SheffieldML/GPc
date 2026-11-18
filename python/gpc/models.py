"""Thin Python model wrappers with paramz-like display (CIP-001E)."""

from __future__ import annotations

from typing import Any, Callable

from ._display import OptimiserTableProgress, format_model_description
from ._gpc import GPRegression as _GPRegression
from ._gpc import OptimiserEvent


class GPRegression(_GPRegression):
    """FTC Gaussian GP; ``str``/``repr`` use a paramz-inspired parameter table."""

    def __str__(self) -> str:  # noqa: D105
        return format_model_description(
            self.describe(),
            name="GP regression",
            objective=self.objective(),
        )

    def __repr__(self) -> str:  # noqa: D105
        return self.__str__()

    def optimize(
        self,
        iters: int = 100,
        messages: bool = False,
        callback: Callable[[OptimiserEvent], Any] | None = None,
    ) -> None:
        """Run the C++ optimiser.

        ``messages=True`` refreshes the paramz-style parameter table in place
        (IPython ``clear_output``, or ANSI rewrite in a plain terminal). It does
        **not** stream C++ ``Iteration:`` / gradient-check dumps.
        """
        observers: list[Callable[[OptimiserEvent], Any]] = []
        if messages:
            observers.append(OptimiserTableProgress(self))
        if callback is not None:
            observers.append(callback)

        def _fanout(event: OptimiserEvent) -> None:
            for obs in observers:
                obs(event)

        # C++ CLI progress / verbosity stay off; Python owns messaging.
        _GPRegression.optimize(
            self,
            iters=iters,
            messages=False,
            callback=_fanout if observers else None,
        )
