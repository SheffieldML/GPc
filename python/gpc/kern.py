"""GPy-like kernel factories for the `gpc` façade (CIP-001D).

These are thin descriptors. Compound ``+`` becomes a parts list that the C++
bindings turn into ``CCmpndKern``. Modelling stays in the core.
"""

from __future__ import annotations

from typing import Any, Sequence


class Kern:
    """Base for façade kernel descriptors (thin handles; maths stay in C++)."""

    def __add__(self, other: object) -> Add:
        """Return a compound kernel (``CCmpndKern`` parts list)."""
        if not isinstance(other, Kern):
            return NotImplemented
        return Add(self, other)

    def __radd__(self, other: object) -> Add:
        if not isinstance(other, Kern):
            return NotImplemented
        return Add(other, self)

    def to_parts(self) -> list[dict[str, Any]]:
        """Serialize to the parts list consumed by the C++ constructor."""
        raise NotImplementedError

    def _flat_leaves(self) -> list[Kern]:
        return [self]


class Add(Kern):
    """Sum of kernels (maps to ``CCmpndKern``)."""

    def __init__(self, *items: Kern) -> None:
        """Build a flattened sum of kernel descriptors."""
        leaves: list[Kern] = []
        for item in items:
            leaves.extend(item._flat_leaves())
        if len(leaves) < 2:
            raise ValueError("Add requires at least two kernel components")
        self._leaves = leaves

    def _flat_leaves(self) -> list[Kern]:
        return list(self._leaves)

    def to_parts(self) -> list[dict[str, Any]]:
        """Serialize each leaf to a parts dict for the C++ binding."""
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
    """Isotropic RBF / squared-exponential (inverseWidth storage; lengthscale paths)."""

    _type = "rbf"

    def __init__(
        self,
        input_dim: int,
        variance: float = 1.0,
        lengthscale: float = 1.0,
    ) -> None:
        """Create an RBF kernel descriptor.

        Parameters
        ----------
        input_dim
            Input dimension.
        variance, lengthscale
            GPy-natural units (CIP-0014 paths ``rbf.variance``, ``rbf.lengthscale``).
        """
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
        """Create an ExpQuad descriptor (paths under ``ExpQuad.*``)."""
        super().__init__(
            input_dim, variance=variance, lengthscale=lengthscale
        )


EQ = ExpQuad


class White(_Atomic):
    """White noise kernel (often used for observation noise in compounds)."""

    _type = "white"

    def __init__(self, input_dim: int, variance: float = 1.0) -> None:
        """Create a White kernel descriptor (``white.variance``)."""
        super().__init__(input_dim, variance=variance)


class Bias(_Atomic):
    """Constant bias kernel."""

    _type = "bias"

    def __init__(self, input_dim: int, variance: float = 1.0) -> None:
        """Create a Bias kernel descriptor (``bias.variance``)."""
        super().__init__(input_dim, variance=variance)


class MLP(_Atomic):
    """Williams infinite-MLP (arcsin) covariance.

    ``weight_variance`` scales ``x`` in the angle features. Large-magnitude
    inputs (calendar years, etc.) saturate unless this is ``O(1/Var(X))``;
    string ``kernel='mlp'`` picks that scale from training ``X``. Explicit
    descriptors default to ``1.0`` (GPy-like); set them from the data scale
    when inputs are not already standardised.
    """

    _type = "mlp"

    def __init__(
        self,
        input_dim: int,
        variance: float = 1.0,
        weight_variance: float = 1.0,
        bias_variance: float = 1.0,
    ) -> None:
        """Create an MLP kernel descriptor (``mlp.*`` paths / aliases)."""
        super().__init__(
            input_dim,
            variance=variance,
            weight_variance=weight_variance,
            bias_variance=bias_variance,
        )


class ArcCosine(_Atomic):
    """Cho–Saul arc-cosine kernel (order 0 or 1)."""

    _type = "arccos"

    def __init__(
        self,
        input_dim: int,
        variance: float = 1.0,
        order: int = 0,
    ) -> None:
        """Create an ArcCosine kernel descriptor (``arccos.variance``; order 0|1)."""
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
