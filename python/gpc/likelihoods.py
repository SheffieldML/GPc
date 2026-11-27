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


class _MultiClass(Likelihood):
    """Block likelihood over ``K`` latent functions per datum (CIP-003E).

    Targets are integer labels ``0..K-1`` or one-hot ``N x K``; the C++ core always
    works with one-hot targets and the conversion happens at this boundary.
    """

    num_classes: int

    def _call_params(self) -> dict[str, Any]:
        raise NotImplementedError

    def _check(self, mu: np.ndarray, var: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mu = np.atleast_2d(np.asarray(mu, dtype=float))
        var = np.atleast_2d(np.asarray(var, dtype=float))
        if mu.shape != var.shape or mu.shape[1] != self.num_classes:
            raise ValueError(f"mu and var must both be N x {self.num_classes}")
        return mu, var

    def expected_loglik(self, y: Any, mu: Any, var: Any) -> np.ndarray:
        """Per-row ``E_q[log p(y|f)]`` with ``q = N(mu, diag(var))`` (C++ oracle)."""
        mu, var = self._check(mu, var)
        return _gpc.multiclass_expected_loglik(
            self.type, np.asarray(y, dtype=float), mu, var, **self._call_params()
        )

    def predict_proba(self, mu: Any, var: Any) -> np.ndarray:
        """Class probabilities (N x K) for marginals ``N(mu, var)`` (C++ oracle)."""
        mu, var = self._check(mu, var)
        return _gpc.multiclass_predict_proba(self.type, mu, var, **self._call_params())

    def _labels(self, y: Any) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        if y.ndim == 2 and y.shape[1] == self.num_classes:
            return np.argmax(y, axis=1)
        lab = y.reshape(-1)
        if np.any(lab != np.floor(lab)) or lab.min() < 0 or lab.max() >= self.num_classes:
            raise ValueError(f"labels must be integers in 0..{self.num_classes - 1}")
        return lab.astype(int)


class RobustMax(_MultiClass):
    """Robust-max multi-class likelihood (Hernandez-Lobato et al., 2011).

    ``p(y=c|f) = 1 - eps`` if ``c`` is the argmax of ``f`` and ``eps / (K - 1)``
    otherwise. ``eps`` is fixed, not learned.
    """

    def __init__(self, num_classes: int, eps: float = 1e-3) -> None:
        if int(num_classes) < 2:
            raise ValueError("num_classes must be at least 2")
        if not 0.0 < float(eps) < 1.0:
            raise ValueError("eps must be in (0, 1)")
        super().__init__("robustmax", num_classes=num_classes, eps=eps)
        self.num_classes = int(num_classes)
        self.eps = float(eps)

    def _call_params(self) -> dict[str, Any]:
        return {"num_classes": self.num_classes, "eps": self.eps}

    def logpdf(self, f: Any, y: Any) -> np.ndarray:
        """Row-wise ``log p(y|f)`` for ``f`` of shape ``N x K``."""
        f = np.atleast_2d(np.asarray(f, dtype=float))
        lab = self._labels(y)
        hit = np.argmax(f, axis=1) == lab
        return np.where(hit, np.log1p(-self.eps), np.log(self.eps / (self.num_classes - 1)))


class Softmax(_MultiClass):
    """Softmax multi-class likelihood; expectations use fixed Monte Carlo draws.

    ``num_samples`` antithetic draws (seeded by ``seed``) are held fixed, so the
    objective is a deterministic function of the variational parameters.
    """

    def __init__(self, num_classes: int, num_samples: int = 256, seed: int = 12345) -> None:
        if int(num_classes) < 2:
            raise ValueError("num_classes must be at least 2")
        if int(num_samples) < 2:
            raise ValueError("num_samples must be at least 2")
        super().__init__("softmax", num_classes=num_classes, num_samples=num_samples, seed=seed)
        self.num_classes = int(num_classes)
        self.num_samples = int(num_samples)
        self.seed = int(seed)

    def _call_params(self) -> dict[str, Any]:
        return {"num_classes": self.num_classes, "samples": self.num_samples, "seed": self.seed}

    def logpdf(self, f: Any, y: Any) -> np.ndarray:
        """Row-wise ``log p(y|f)`` for ``f`` of shape ``N x K``."""
        f = np.atleast_2d(np.asarray(f, dtype=float))
        lab = self._labels(y)
        m = f.max(axis=1, keepdims=True)
        lse = m[:, 0] + np.log(np.exp(f - m).sum(axis=1))
        return f[np.arange(f.shape[0]), lab] - lse


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
    if t in ("robustmax", "robust_max"):
        return RobustMax(num_classes=int(d["num_classes"]), eps=float(d.get("eps", 1e-3)))
    if t == "softmax":
        return Softmax(
            num_classes=int(d["num_classes"]),
            num_samples=int(d.get("num_samples", 256)),
            seed=int(d.get("seed", 12345)),
        )
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
    "RobustMax",
    "Softmax",
    "from_dict",
]
