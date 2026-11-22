"""GPy-facing observation likelihood constructors (CIP-003A / REQ-001B).

Thin façades over C++ ``CNoise`` families. Densities evaluate in the C++ core
via ``gpc._gpc.noise_log_density``.

Example::

    import gpc
    lik = gpc.likelihoods.SkewGaussian(variance=0.1, alpha=2.0)
    print(lik.logpdf(f=0.0, y=0.5))
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from . import _gpc


def _as1d(a: Any) -> np.ndarray:
    return np.asarray(a, dtype=float).reshape(-1)


class Likelihood:
    """Base: stream ``type`` string plus parameter dict for C++ oracles."""

    type: str = "gaussian"
    params: dict[str, float]

    def __init__(self, type: str = "gaussian", **params: float) -> None:
        self.type = type
        self.params = {k: float(v) for k, v in params.items()}

    def logpdf(self, f: Any, y: Any) -> np.ndarray:
        """Elementwise ``log p(y|f)`` via the C++ Laplace density oracle."""
        return _gpc.noise_log_density(self.type, _as1d(y), _as1d(f), self.params)

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, **self.params}


class Gaussian(Likelihood):
    def __init__(self, variance: float = 1.0) -> None:
        super().__init__("gaussian", variance=variance)
        self.variance = float(variance)


class Bernoulli(Likelihood):
    """Bernoulli with probit link (GPy Bernoulli / GPc ``CProbitNoise``)."""

    def __init__(self) -> None:
        super().__init__("probit")


class SkewGaussian(Likelihood):
    def __init__(self, variance: float = 1.0, alpha: float = 0.0) -> None:
        super().__init__("skew_gaussian", variance=variance, alpha=alpha)
        self.variance = float(variance)
        self.alpha = float(alpha)


class SkewExponential(Likelihood):
    def __init__(self, rate: float = 1.0, alpha: float = 0.0) -> None:
        super().__init__("skew_exponential", rate=rate, alpha=alpha)
        self.rate = float(rate)
        self.alpha = float(alpha)


class StudentT(Likelihood):
    def __init__(self, variance: float = 1.0, deg_free: float = 5.0) -> None:
        super().__init__("student_t", variance=variance, deg_free=deg_free)
        self.variance = float(variance)
        self.deg_free = float(deg_free)


class Poisson(Likelihood):
    def __init__(self) -> None:
        super().__init__("poisson")


class Binomial(Likelihood):
    def __init__(self, num_trials: int = 1) -> None:
        super().__init__("binomial", num_trials=float(num_trials))
        self.num_trials = int(num_trials)


class Exponential(Likelihood):
    def __init__(self) -> None:
        super().__init__("exponential")


class Gamma(Likelihood):
    def __init__(self, beta: float = 1.0) -> None:
        super().__init__("gamma", beta=beta)
        self.beta = float(beta)


def from_dict(d: Mapping[str, Any]) -> Likelihood:
    """Build a likelihood from a ``{type, ...params}`` mapping."""
    t = str(d.get("type", "gaussian")).lower()
    if t == "gaussian":
        return Gaussian(variance=float(d.get("variance", 1.0)))
    if t in ("probit", "bernoulli"):
        return Bernoulli()
    if t in ("skew_gaussian", "skewgaussian"):
        return SkewGaussian(
            variance=float(d.get("variance", 1.0)),
            alpha=float(d.get("alpha", 0.0)),
        )
    if t in ("skew_exponential", "skewexponential"):
        return SkewExponential(
            rate=float(d.get("rate", 1.0)),
            alpha=float(d.get("alpha", 0.0)),
        )
    if t in ("student_t", "studentt"):
        return StudentT(
            variance=float(d.get("variance", 1.0)),
            deg_free=float(d.get("deg_free", 5.0)),
        )
    if t == "poisson":
        return Poisson()
    if t == "binomial":
        return Binomial(num_trials=int(d.get("num_trials", 1)))
    if t == "exponential":
        return Exponential()
    if t == "gamma":
        return Gamma(beta=float(d.get("beta", 1.0)))
    raise ValueError(f"Unknown likelihood type {t!r}")


__all__ = [
    "Likelihood",
    "Gaussian",
    "Bernoulli",
    "SkewGaussian",
    "SkewExponential",
    "StudentT",
    "Poisson",
    "Binomial",
    "Exponential",
    "Gamma",
    "from_dict",
]
