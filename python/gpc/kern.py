"""GPy-like kernel factories for the `gpc` façade (CIP-001D).

These are thin descriptors. Compound ``+`` becomes a parts list that the C++
bindings turn into ``CCmpndKern``. Modelling stays in the core.
"""

from __future__ import annotations

from typing import Any, Sequence


class Kern:
    """Base for façade kernel descriptors."""

    def __add__(self, other: object) -> Add:
        if not isinstance(other, Kern):
            return NotImplemented
        return Add(self, other)

    def __radd__(self, other: object) -> Add:
        if not isinstance(other, Kern):
            return NotImplemented
        return Add(other, self)

    def to_parts(self) -> list[dict[str, Any]]:
        raise NotImplementedError

    def _flat_leaves(self) -> list[Kern]:
        return [self]


class Add(Kern):
    """Sum of kernels (maps to ``CCmpndKern``)."""

    def __init__(self, *items: Kern) -> None:
        leaves: list[Kern] = []
        for item in items:
            leaves.extend(item._flat_leaves())
        if len(leaves) < 2:
            raise ValueError("Add requires at least two kernel components")
        self._leaves = leaves

    def _flat_leaves(self) -> list[Kern]:
        return list(self._leaves)

    def to_parts(self) -> list[dict[str, Any]]:
        return [leaf.to_parts()[0] for leaf in self._leaves]

    def __repr__(self) -> str:
        return " + ".join(repr(leaf) for leaf in self._leaves)


class _Atomic(Kern):
    _type: str = ""

    def __init__(self, input_dim: int, **params: float) -> None:
        if int(input_dim) < 1:
            raise ValueError("input_dim must be >= 1")
        self.input_dim = int(input_dim)
        self.params = {k: float(v) for k, v in params.items()}

    def to_parts(self) -> list[dict[str, Any]]:
        part: dict[str, Any] = {"type": self._type, "input_dim": self.input_dim}
        part.update(self.params)
        return [part]

    def __repr__(self) -> str:
        extras = ", ".join(f"{k}={v!r}" for k, v in self.params.items())
        if extras:
            return f"{type(self).__name__}({self.input_dim}, {extras})"
        return f"{type(self).__name__}({self.input_dim})"


class RBF(_Atomic):
    _type = "rbf"

    def __init__(
        self,
        input_dim: int,
        variance: float = 1.0,
        lengthscale: float = 1.0,
    ) -> None:
        super().__init__(
            input_dim, variance=variance, lengthscale=lengthscale
        )


class ExpQuad(_Atomic):
    """GPy ExpQuad / EQ lengthscale-first squared exponential (CIP-0024)."""

    _type = "eq"

    def __init__(
        self,
        input_dim: int,
        variance: float = 1.0,
        lengthscale: float = 1.0,
    ) -> None:
        super().__init__(
            input_dim, variance=variance, lengthscale=lengthscale
        )


EQ = ExpQuad


class White(_Atomic):
    _type = "white"

    def __init__(self, input_dim: int, variance: float = 1.0) -> None:
        super().__init__(input_dim, variance=variance)


class Bias(_Atomic):
    _type = "bias"

    def __init__(self, input_dim: int, variance: float = 1.0) -> None:
        super().__init__(input_dim, variance=variance)


class MLP(_Atomic):
    _type = "mlp"

    def __init__(
        self,
        input_dim: int,
        variance: float = 1.0,
        weight_variance: float = 10.0,
        bias_variance: float = 10.0,
    ) -> None:
        super().__init__(
            input_dim,
            variance=variance,
            weight_variance=weight_variance,
            bias_variance=bias_variance,
        )


class ArcCosine(_Atomic):
    _type = "arccos"

    def __init__(
        self,
        input_dim: int,
        variance: float = 1.0,
        order: int = 0,
    ) -> None:
        if int(order) not in (0, 1):
            raise ValueError("ArcCosine order must be 0 or 1")
        super().__init__(input_dim, variance=variance)
        self.params["order"] = float(int(order))


def normalize_kernel_arg(kernel: object) -> str | list[dict[str, Any]]:
    """Return a string name or a parts list for the C++ constructor."""
    if isinstance(kernel, str):
        return kernel
    if isinstance(kernel, Kern):
        return kernel.to_parts()
    if isinstance(kernel, Sequence) and not isinstance(kernel, (str, bytes)):
        # Already a parts list of dicts
        return list(kernel)  # type: ignore[arg-type]
    raise TypeError(
        "kernel must be a string name, a gpc.kern.Kern, or a parts list; "
        f"got {type(kernel)!r}"
    )


__all__ = [
    "Kern",
    "Add",
    "RBF",
    "ExpQuad",
    "EQ",
    "White",
    "Bias",
    "MLP",
    "ArcCosine",
    "normalize_kernel_arg",
]
