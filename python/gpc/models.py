"""Thin Python model wrappers with paramz-like display (CIP-001E)."""

from __future__ import annotations

from ._display import format_model_description
from ._gpc import GPRegression as _GPRegression


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
