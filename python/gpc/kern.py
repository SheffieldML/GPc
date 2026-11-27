"""GPy-like kernel factories for the `gpc` façade (CIP-001D / CIP-0038).

These are thin descriptors. Compound ``+`` becomes a parts list that the C++
bindings turn into ``CCmpndKern``; product ``*`` becomes nested ``tensor``
parts (``CTensorKern``). Modelling stays in the core.

Naming layers (CIP-0014 / CIP-0038):
  - Python class: GPy constructor name (``Matern32``, ``StdPeriodic``, …)
  - Parts ``type``: GPc stream string (``matern32``, ``stdperiodic``, …)
  - Paths after build: GPy/paramz segment (``Mat32``, ``std_periodic``, …)
"""

from __future__ import annotations

from typing import Any, Sequence

import copy


class Kern:
    """Base for façade kernel descriptors (thin handles; maths stay in C++)."""

    active_dims: list[int] | None = None

    def with_active_dims(self, dims: Sequence[int]) -> Kern:
        """Return a shallow copy with GPy-style ``active_dims`` set."""
        out = copy.copy(self)
        out.active_dims = [int(d) for d in dims]
        return out

    def __add__(self, other: object) -> Add:
        """Return a compound kernel (``CCmpndKern`` parts list)."""
        if not isinstance(other, Kern):
            return NotImplemented
        return Add(self, other)

    def __radd__(self, other: object) -> Add:
        if not isinstance(other, Kern):
            return NotImplemented
        return Add(other, self)

    def __mul__(self, other: object) -> Prod:
        """Return a product kernel (``CTensorKern`` / GPy ``Prod``)."""
        if not isinstance(other, Kern):
            return NotImplemented
        return Prod(self, other)

    def __rmul__(self, other: object) -> Prod:
        if not isinstance(other, Kern):
            return NotImplemented
        return Prod(other, self)

    def to_parts(self) -> list[dict[str, Any]]:
        """Serialize to the parts list consumed by the C++ constructor."""
        raise NotImplementedError

    def K(
        self,
        X: Any,
        X2: Any | None = None,
    ) -> Any:
        """Evaluate the covariance matrix ``K(X)`` or ``K(X, X2)``.

        Builds a temporary ``CKern`` from ``to_parts()`` (no GP / likelihood).
        Returns a NumPy ``(n, n)`` or ``(n, n2)`` array. Honours ``active_dims``.
        """
        import numpy as np

        from ._gpc import kern_K

        Xa = np.asarray(X, dtype=float)
        if Xa.ndim == 1:
            Xa = Xa[:, None]
        X2a = None
        if X2 is not None:
            X2a = np.asarray(X2, dtype=float)
            if X2a.ndim == 1:
                X2a = X2a[:, None]
        return kern_K(self.to_parts(), Xa, X2a)

    def plot(self, *args: Any, **kwargs: Any) -> Any:
        """Thin GPy-style kernel plot (requires ``gpc[plot]`` / matplotlib)."""
        from .plot import plot_kern

        return plot_kern(self, *args, **kwargs)

    def _flat_sum_leaves(self) -> list[Kern]:
        """Leaves for sum flattening (``Add`` only); products stay intact."""
        return [self]

    def _flat_prod_factors(self) -> list[Kern]:
        """Factors for product flattening (``Prod`` only); sums stay intact."""
        return [self]


class Add(Kern):
    """Sum of kernels (maps to ``CCmpndKern``)."""

    def __init__(self, *items: Kern) -> None:
        """Build a flattened sum of kernel descriptors."""
        leaves: list[Kern] = []
        for item in items:
            leaves.extend(item._flat_sum_leaves())
        if len(leaves) < 2:
            raise ValueError("Add requires at least two kernel components")
        self._leaves = leaves

    def _flat_sum_leaves(self) -> list[Kern]:
        return list(self._leaves)

    def to_parts(self) -> list[dict[str, Any]]:
        """Serialize each summand (atomic or nested tensor/cmpnd)."""
        out: list[dict[str, Any]] = []
        for leaf in self._leaves:
            out.extend(leaf.to_parts())
        return out

    def __repr__(self) -> str:
        return " + ".join(repr(leaf) for leaf in self._leaves)


class Prod(Kern):
    """Product of kernels (maps to ``CTensorKern``; GPy ``Prod``).

    White noise must be summed outside the product (``k * lin + White``), not
    multiplied in — matching ``CTensorKern`` / CIP-000D.
    """

    def __init__(self, *items: Kern) -> None:
        """Build a flattened product of kernel descriptors."""
        factors: list[Kern] = []
        for item in items:
            factors.extend(item._flat_prod_factors())
        if len(factors) < 2:
            raise ValueError("Prod requires at least two kernel factors")
        self._factors = factors

    def _flat_prod_factors(self) -> list[Kern]:
        return list(self._factors)

    def _factor_part(self, factor: Kern) -> dict[str, Any]:
        parts = factor.to_parts()
        if isinstance(factor, Add):
            return {"type": "cmpnd", "parts": parts}
        if len(parts) != 1:
            raise ValueError(
                "Prod factors must serialize to one part dict "
                f"(got {len(parts)} from {type(factor).__name__})"
            )
        return parts[0]

    def to_parts(self) -> list[dict[str, Any]]:
        """One nested ``tensor`` part for the C++ binding."""
        return [
            {
                "type": "tensor",
                "parts": [self._factor_part(f) for f in self._factors],
            }
        ]

    def __repr__(self) -> str:
        return " * ".join(repr(f) for f in self._factors)


class _Atomic(Kern):
    _type: str = ""
    _ard_type: str | None = None

    def __init__(
        self,
        input_dim: int,
        *,
        ARD: bool = False,
        active_dims: Sequence[int] | None = None,
        **params: float,
    ) -> None:
        if int(input_dim) < 1:
            raise ValueError("input_dim must be >= 1")
        self.input_dim = int(input_dim)
        self.ARD = bool(ARD)
        if self.ARD and self._ard_type is None:
            raise ValueError(f"{type(self).__name__} does not support ARD=True")
        self.active_dims = (
            [int(d) for d in active_dims] if active_dims is not None else None
        )
        self.params = {k: float(v) for k, v in params.items()}

    def _stream_type(self) -> str:
        if self.ARD:
            assert self._ard_type is not None
            return self._ard_type
        return self._type

    def to_parts(self) -> list[dict[str, Any]]:
        part: dict[str, Any] = {
            "type": self._stream_type(),
            "input_dim": self.input_dim,
            "ARD": self.ARD,
        }
        part.update(self.params)
        if self.active_dims is not None:
            part["active_dims"] = list(self.active_dims)
        return [part]

    def __repr__(self) -> str:
        extras = ", ".join(f"{k}={v!r}" for k, v in self.params.items())
        ard = ", ARD=True" if self.ARD else ""
        dims = (
            f", active_dims={self.active_dims!r}"
            if self.active_dims is not None
            else ""
        )
        if extras:
            return f"{type(self).__name__}({self.input_dim}, {extras}{ard}{dims})"
        return f"{type(self).__name__}({self.input_dim}{ard}{dims})"


class RBF(_Atomic):
    """Isotropic RBF / squared-exponential (inverseWidth storage; lengthscale paths)."""

    _type = "rbf"
    _ard_type = "rbfard"

    def __init__(
        self,
        input_dim: int,
        variance: float = 1.0,
        lengthscale: float = 1.0,
        ARD: bool = False,
        active_dims: Sequence[int] | None = None,
    ) -> None:
        """Create an RBF kernel descriptor.

        Parameters
        ----------
        input_dim
            Input dimension.
        variance, lengthscale
            GPy-natural units (CIP-0014 paths ``rbf.variance``, ``rbf.lengthscale``).
        ARD
            If True, use ``rbfard`` (per-dimension input scales).
        """
        super().__init__(
            input_dim,
            ARD=ARD,
            variance=variance,
            lengthscale=lengthscale,
            active_dims=active_dims,
        )


class ExpQuad(_Atomic):
    """GPy ExpQuad / EQ lengthscale-first squared exponential (CIP-0024)."""

    _type = "eq"
    _ard_type = None

    def __init__(
        self,
        input_dim: int,
        variance: float = 1.0,
        lengthscale: float = 1.0,
        ARD: bool = False,
        active_dims: Sequence[int] | None = None,
    ) -> None:
        """Create an ExpQuad descriptor (paths under ``ExpQuad.*``)."""
        if ARD:
            raise ValueError("ExpQuad ARD is not available; use RBF(..., ARD=True)")
        super().__init__(
            input_dim,
            variance=variance,
            lengthscale=lengthscale,
            active_dims=active_dims,
        )


EQ = ExpQuad


class White(_Atomic):
    """White noise kernel (often used for observation noise in compounds)."""

    _type = "white"

    def __init__(
        self,
        input_dim: int,
        variance: float = 1.0,
        active_dims: Sequence[int] | None = None,
    ) -> None:
        """Create a White kernel descriptor (``white.variance``)."""
        super().__init__(input_dim, variance=variance, active_dims=active_dims)


class Bias(_Atomic):
    """Constant bias kernel."""

    _type = "bias"

    def __init__(
        self,
        input_dim: int,
        variance: float = 1.0,
        active_dims: Sequence[int] | None = None,
    ) -> None:
        """Create a Bias kernel descriptor (``bias.variance``)."""
        super().__init__(input_dim, variance=variance, active_dims=active_dims)


class MLP(_Atomic):
    """Williams infinite-MLP (arcsin) covariance (GPy ``kern.MLP`` defaults)."""

    _type = "mlp"
    _ard_type = "mlpard"

    def __init__(
        self,
        input_dim: int,
        variance: float = 1.0,
        weight_variance: float = 1.0,
        bias_variance: float = 1.0,
        ARD: bool = False,
        active_dims: Sequence[int] | None = None,
    ) -> None:
        """Create an MLP kernel descriptor (``mlp.*`` paths / aliases).

        Defaults match GPy: ``variance=weight_variance=bias_variance=1``.
        Large-magnitude inputs need an explicit ``weight_variance`` (e.g.
        ``1/Var(X)``) or standardised ``X`` — see ``mlp_olympic.ipynb``.
        """
        super().__init__(
            input_dim,
            ARD=ARD,
            variance=variance,
            weight_variance=weight_variance,
            bias_variance=bias_variance,
            active_dims=active_dims,
        )


class ArcCosine(_Atomic):
    """Cho–Saul arc-cosine kernel (order 0 or 1)."""

    _type = "arccos"

    def __init__(
        self,
        input_dim: int,
        variance: float = 1.0,
        order: int = 0,
        active_dims: Sequence[int] | None = None,
    ) -> None:
        """Create an ArcCosine kernel descriptor (``arccos.variance``; order 0|1)."""
        if int(order) not in (0, 1):
            raise ValueError("ArcCosine order must be 0 or 1")
        super().__init__(input_dim, variance=variance, active_dims=active_dims)
        self.params["order"] = float(int(order))


class Matern32(_Atomic):
    """Matérn 3/2 (GPy ``Matern32``; path segment ``Mat32``)."""

    _type = "matern32"
    _ard_type = "matern32ard"

    def __init__(
        self,
        input_dim: int,
        variance: float = 1.0,
        lengthscale: float = 1.0,
        ARD: bool = False,
        active_dims: Sequence[int] | None = None,
    ) -> None:
        """Create a Matérn 3/2 descriptor."""
        super().__init__(
            input_dim,
            ARD=ARD,
            variance=variance,
            lengthscale=lengthscale,
            active_dims=active_dims,
        )


Mat32 = Matern32


class Matern52(_Atomic):
    """Matérn 5/2 (GPy ``Matern52``; path segment ``Mat52``)."""

    _type = "matern52"
    _ard_type = "matern52ard"

    def __init__(
        self,
        input_dim: int,
        variance: float = 1.0,
        lengthscale: float = 1.0,
        ARD: bool = False,
        active_dims: Sequence[int] | None = None,
    ) -> None:
        """Create a Matérn 5/2 descriptor."""
        super().__init__(
            input_dim,
            ARD=ARD,
            variance=variance,
            lengthscale=lengthscale,
            active_dims=active_dims,
        )


Mat52 = Matern52


class Exponential(_Atomic):
    """Exponential / Ornstein–Uhlenbeck (GPy ``Exponential``; ``OU`` is an alias)."""

    _type = "exp"

    def __init__(
        self,
        input_dim: int,
        variance: float = 1.0,
        lengthscale: float = 1.0,
        active_dims: Sequence[int] | None = None,
    ) -> None:
        """Create an Exponential descriptor (paths ``Exponential.*``)."""
        super().__init__(
            input_dim,
            variance=variance,
            lengthscale=lengthscale,
            active_dims=active_dims,
        )


OU = Exponential


class RatQuad(_Atomic):
    """Rational quadratic (GPy ``RatQuad``; leaf ``power`` aliases stored ``alpha``)."""

    _type = "ratquad"
    _ard_type = "ratquadard"

    def __init__(
        self,
        input_dim: int,
        variance: float = 1.0,
        lengthscale: float = 1.0,
        power: float = 2.0,
        ARD: bool = False,
        active_dims: Sequence[int] | None = None,
    ) -> None:
        """Create a RatQuad descriptor."""
        super().__init__(
            input_dim,
            ARD=ARD,
            variance=variance,
            lengthscale=lengthscale,
            power=power,
            active_dims=active_dims,
        )


class StdPeriodic(_Atomic):
    """Standard periodic (GPy ``StdPeriodic``; path segment ``std_periodic``)."""

    _type = "stdperiodic"

    def __init__(
        self,
        input_dim: int,
        variance: float = 1.0,
        period: float = 1.0,
        lengthscale: float = 1.0,
        active_dims: Sequence[int] | None = None,
    ) -> None:
        """Create a StdPeriodic descriptor."""
        super().__init__(
            input_dim,
            variance=variance,
            period=period,
            lengthscale=lengthscale,
            active_dims=active_dims,
        )


class Cosine(_Atomic):
    """Cosine stationary kernel (GPy ``Cosine``)."""

    _type = "cosine"

    def __init__(
        self,
        input_dim: int,
        variance: float = 1.0,
        lengthscale: float = 1.0,
        active_dims: Sequence[int] | None = None,
    ) -> None:
        """Create a Cosine descriptor."""
        super().__init__(
            input_dim,
            variance=variance,
            lengthscale=lengthscale,
            active_dims=active_dims,
        )


class Brownian(_Atomic):
    """Brownian motion on a 1-D index (GPy ``Brownian``)."""

    _type = "brownian"

    def __init__(
        self,
        input_dim: int = 1,
        variance: float = 1.0,
        active_dims: Sequence[int] | None = None,
    ) -> None:
        """Create a Brownian descriptor (requires ``input_dim == 1``)."""
        if int(input_dim) != 1:
            raise ValueError("Brownian requires input_dim == 1")
        super().__init__(input_dim, variance=variance, active_dims=active_dims)


class Linear(_Atomic):
    """Linear kernel (GPy ``Linear``; path segment ``linear``)."""

    _type = "lin"
    _ard_type = "linard"

    def __init__(
        self,
        input_dim: int,
        variance: float = 1.0,
        ARD: bool = False,
        active_dims: Sequence[int] | None = None,
    ) -> None:
        """Create a Linear descriptor."""
        super().__init__(
            input_dim, ARD=ARD, variance=variance, active_dims=active_dims
        )


class Poly(_Atomic):
    """Polynomial kernel (GPy ``Poly``)."""

    _type = "poly"
    _ard_type = "polyard"

    def __init__(
        self,
        input_dim: int,
        variance: float = 1.0,
        weight_variance: float = 1.0,
        bias_variance: float = 1.0,
        degree: float = 2.0,
        ARD: bool = False,
        active_dims: Sequence[int] | None = None,
    ) -> None:
        """Create a Poly descriptor (``degree`` is integer-valued)."""
        super().__init__(
            input_dim,
            ARD=ARD,
            variance=variance,
            weight_variance=weight_variance,
            bias_variance=bias_variance,
            degree=float(int(degree)),
            active_dims=active_dims,
        )


class Sinc(_Atomic):
    """Sinc stationary kernel (GPy ``Sinc``; CIP-0038 Phase B)."""

    _type = "sinc"

    def __init__(
        self,
        input_dim: int,
        variance: float = 1.0,
        lengthscale: float = 1.0,
        active_dims: Sequence[int] | None = None,
    ) -> None:
        """Create a Sinc descriptor (``K(r)=variance*sinc(2r)`` in numpy sense)."""
        super().__init__(
            input_dim,
            variance=variance,
            lengthscale=lengthscale,
            active_dims=active_dims,
        )


class ExpQuadCosine(_Atomic):
    """ExpQuad × Cosine spectral-mixture style (GPy ``ExpQuadCosine``; CIP-0038)."""

    _type = "expquadcosine"

    def __init__(
        self,
        input_dim: int,
        variance: float = 1.0,
        lengthscale: float = 1.0,
        period: float = 1.0,
        active_dims: Sequence[int] | None = None,
    ) -> None:
        """Create an ExpQuadCosine descriptor."""
        super().__init__(
            input_dim,
            variance=variance,
            lengthscale=lengthscale,
            period=period,
            active_dims=active_dims,
        )


class Coregionalize(Kern):
    """Output coregionalization B = W W^T + diag(kappa) (GPy ``Coregionalize``).

    Multiply with a spatial kernel for ICM: ``RBF(q) * Coregionalize(1, D)`` with
    matching ``active_dims``, or use ``gpc.util.multioutput.ICM``.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        rank: int = 1,
        *,
        active_dims: Sequence[int] | None = None,
    ) -> None:
        """Create a Coregionalize descriptor (``input_dim`` is usually 1)."""
        if int(input_dim) < 1:
            raise ValueError("input_dim must be >= 1")
        if int(output_dim) < 1:
            raise ValueError("output_dim must be >= 1")
        if int(rank) < 1:
            raise ValueError("rank must be >= 1")
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.rank = int(rank)
        self.active_dims = (
            [int(d) for d in active_dims] if active_dims is not None else None
        )

    def to_parts(self) -> list[dict[str, Any]]:
        part: dict[str, Any] = {
            "type": "coregionalize",
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "rank": self.rank,
        }
        if self.active_dims is not None:
            part["active_dims"] = list(self.active_dims)
        return [part]

    def __repr__(self) -> str:
        dims = (
            f", active_dims={self.active_dims!r}"
            if self.active_dims is not None
            else ""
        )
        return (
            f"Coregionalize({self.input_dim}, output_dim={self.output_dim}, "
            f"rank={self.rank}{dims})"
        )


class IndexWhite(Kern):
    """Per-output diagonal noise on the index column (GPc ``indexwhite`` / CIP-000D).

    Sum outside an ICM/LCM tensor — never multiply inside ``Prod``.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        variance: float | None = None,
        variances: Sequence[float] | None = None,
        active_dims: Sequence[int] | None = None,
    ) -> None:
        """Create an IndexWhite descriptor (``input_dim`` is usually 1)."""
        if int(input_dim) < 1:
            raise ValueError("input_dim must be >= 1")
        if int(output_dim) < 1:
            raise ValueError("output_dim must be >= 1")
        if variance is not None and variances is not None:
            raise ValueError("pass only one of variance= or variances=")
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.variance = float(variance) if variance is not None else None
        self.variances = (
            [float(v) for v in variances] if variances is not None else None
        )
        if self.variances is not None and len(self.variances) != self.output_dim:
            raise ValueError("variances length must equal output_dim")
        self.active_dims = (
            [int(d) for d in active_dims] if active_dims is not None else None
        )

    def to_parts(self) -> list[dict[str, Any]]:
        part: dict[str, Any] = {
            "type": "indexwhite",
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
        }
        if self.variances is not None:
            part["variances"] = list(self.variances)
        elif self.variance is not None:
            part["variance"] = self.variance
        if self.active_dims is not None:
            part["active_dims"] = list(self.active_dims)
        return [part]

    def __repr__(self) -> str:
        dims = (
            f", active_dims={self.active_dims!r}"
            if self.active_dims is not None
            else ""
        )
        return f"IndexWhite({self.input_dim}, output_dim={self.output_dim}{dims})"


class Conv(Kern):
    """Convolutional covariance on flattened images (CIP-0040, ``CConvKern``).

    ``k(x, x') = sum_{p,p'} w_p w_{p'} k_g(x[p], x'[p'])`` where ``x[p]`` is the
    ``p``-th patch and ``k_g`` is the patch kernel ``base``. Images are flattened
    row-major with the channel fastest (``H x W x C``); patches are the valid
    windows at ``stride``, flattened in the same ``(h, w, C)`` order.

    Parameters
    ----------
    base :
        Patch kernel: :class:`RBF`, :class:`MLP` or :class:`Linear` with
        ``input_dim == patch_h * patch_w * C``.
    image_shape :
        ``(H, W, C)``.
    patch_shape :
        ``(h, w)``.
    stride :
        Patch stride (default 1).
    weights :
        ``'learned'`` (one weight per patch position), ``'tied'`` (one shared
        weight) or ``'fixed'`` (``w = 1 / P``, no parameters).

    Use it as ``DeepGP(X, Y, kernel=[Conv(...), 'rbf'], inducing='patches')``;
    the model's inducing variables are then patches, and
    ``model.inducing_filters()`` returns them as images. It is not accepted by
    the ``CGp``-based models (``SparseGPRegression`` and friends): use
    ``DeepGP(layers=1)``, with ``likelihood='probit'`` for classification.
    """

    def __init__(
        self,
        base: Kern,
        image_shape: Sequence[int],
        patch_shape: Sequence[int],
        stride: int = 1,
        weights: str = "learned",
    ) -> None:
        """Create a convolutional kernel descriptor."""
        if not isinstance(base, _Atomic):
            raise TypeError("Conv base must be an atomic kernel (RBF, MLP or Linear)")
        if base.active_dims is not None:
            raise ValueError("Conv base must not use active_dims")
        if base._type not in ("rbf", "mlp", "lin", "linear"):
            raise ValueError(
                f"Conv base must be RBF, MLP or Linear (got {type(base).__name__})"
            )
        shape = [int(v) for v in image_shape]
        pshape = [int(v) for v in patch_shape]
        if len(shape) != 3 or len(pshape) != 2:
            raise ValueError("image_shape must be (H, W, C) and patch_shape (h, w)")
        if min(shape + pshape) < 1 or int(stride) < 1:
            raise ValueError("image_shape, patch_shape and stride must be positive")
        if pshape[0] > shape[0] or pshape[1] > shape[1]:
            raise ValueError("patch_shape must fit inside image_shape")
        if weights not in ("learned", "tied", "fixed"):
            raise ValueError("weights must be 'learned', 'tied' or 'fixed'")
        q = pshape[0] * pshape[1] * shape[2]
        if base.input_dim != q:
            raise ValueError(
                f"base.input_dim is {base.input_dim} but a patch has "
                f"h*w*C = {q} entries"
            )
        self.base = base
        self.image_shape = tuple(shape)
        self.patch_shape = tuple(pshape)
        self.stride = int(stride)
        self.weights = weights
        self.input_dim = shape[0] * shape[1] * shape[2]

    @property
    def num_patches(self) -> int:
        """Number of patch positions ``P``."""
        H, W, _ = self.image_shape
        h, w = self.patch_shape
        return ((H - h) // self.stride + 1) * ((W - w) // self.stride + 1)

    def to_parts(self) -> list[dict[str, Any]]:
        """One ``conv`` part (consumed by ``DeepGP`` and ``Conv.K``)."""
        return [
            {
                "type": "conv",
                "input_dim": self.input_dim,
                "image_shape": list(self.image_shape),
                "patch_shape": list(self.patch_shape),
                "stride": self.stride,
                "weights": self.weights,
                "patch": self.base.to_parts()[0],
            }
        ]

    def K(self, X: Any, X2: Any | None = None) -> Any:
        """Evaluate the convolutional covariance ``K(X)`` or ``K(X, X2)``."""
        import numpy as np

        from ._gpc import conv_kern_K

        Xa = np.ascontiguousarray(X, dtype=float)
        X2a = None if X2 is None else np.ascontiguousarray(X2, dtype=float)
        return conv_kern_K(self.to_parts()[0], Xa, X2a)

    def __repr__(self) -> str:
        return (
            f"Conv({self.base!r}, image_shape={self.image_shape}, "
            f"patch_shape={self.patch_shape}, stride={self.stride}, "
            f"weights={self.weights!r})"
        )


def normalize_kernel_arg(kernel: object) -> str | list[dict[str, Any]]:
    """Return a string name or a parts list for the C++ constructor."""
    if isinstance(kernel, str):
        return kernel
    if isinstance(kernel, Conv) or (
        isinstance(kernel, Sequence)
        and not isinstance(kernel, (str, bytes))
        and any(isinstance(p, dict) and p.get("type") == "conv" for p in kernel)
    ):
        raise NotImplementedError(
            "Conv kernels are supported by DeepGP only (use DeepGP(layers=1, "
            "kernel=Conv(...), inducing='patches'); add likelihood='probit' for "
            "classification)"
        )
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
    "Conv",
    "Add",
    "Prod",
    "RBF",
    "ExpQuad",
    "EQ",
    "White",
    "Bias",
    "MLP",
    "ArcCosine",
    "Matern32",
    "Mat32",
    "Matern52",
    "Mat52",
    "Exponential",
    "OU",
    "RatQuad",
    "StdPeriodic",
    "Cosine",
    "Brownian",
    "Linear",
    "Poly",
    "Sinc",
    "ExpQuadCosine",
    "Coregionalize",
    "IndexWhite",
    "normalize_kernel_arg",
]
