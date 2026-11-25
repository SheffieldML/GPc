"""Posterior sampling of GP hyperparameters (CIP-003B / REQ-001C).

Thin façade over the C++ ``CHmc`` sampler. Sampling runs in C++; NumPy arrays
cross the boundary once, after the chain finishes::

    import gpc
    m = gpc.models.GPRegression(X, Y, kernel=gpc.kern.RBF(1))
    for path in ("rbf.lengthscale", "rbf.variance", "likelihood.variance"):
        gpc.inference.set_prior(m, path, "gamma", 2.0, 1.0)
    res = gpc.inference.hmc(m, num_samples=2000, burn_in=300)
    res.to_csv("samples.csv")

Priors sit on the natural (constrained) value of a parameter, as in GPy. Unlike
GPy, the target includes the change-of-variables Jacobian by default
(``include_jacobian=False`` reproduces GPy) and an improper target (a sampled
parameter with no prior) is refused rather than sampled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

_FAMILIES = {"gamma": 2, "gaussian": 1}


def prior_spec(family: str, *params: float) -> str:
    """Return the ``family:p1,p2`` text spec understood by the C++ core.

    ``gamma`` takes ``(shape, rate)``; ``gaussian`` takes ``(precision,)``
    (zero mean). Both are validated in C++ as well.
    """
    key = family.lower()
    if key not in _FAMILIES:
        raise ValueError(
            f"unknown prior family {family!r}; use one of {sorted(_FAMILIES)}"
        )
    if len(params) != _FAMILIES[key]:
        raise ValueError(
            f"{key} prior takes {_FAMILIES[key]} parameter(s), got {len(params)}"
        )
    return key + ":" + ",".join(repr(float(p)) for p in params)


def set_prior(model: Any, path: str, family: str, *params: float) -> None:
    """Attach a prior to the parameter at ``path`` (natural units).

    ``path`` is a GPy-like path such as ``rbf.lengthscale`` or
    ``likelihood.variance``. Supported by ``GPRegression``,
    ``SparseGPRegression``, ``BayesianGPLVM`` (kernel paths and
    ``likelihood.variance``) and ``MRD`` (``view0.lengthscale`` and so on).
    """
    model.set_prior(path, prior_spec(family, *params))


@dataclass
class HmcResult:
    """Output of :func:`hmc`.

    ``samples`` has one row per draw and one column per sampled parameter, in
    natural (constrained) units; ``names`` labels the columns.
    """

    names: list[str]
    samples: np.ndarray
    log_posterior: np.ndarray
    accept_rate: float
    step_size: float

    def to_csv(self, path: str) -> None:
        """Write ``iteration,log_posterior,<names>`` (same layout as ``gp sample``)."""
        header = ",".join(["iteration", "log_posterior", *self.names])
        rows = np.column_stack(
            [
                np.arange(self.samples.shape[0]),
                self.log_posterior.reshape(-1),
                self.samples,
            ]
        )
        np.savetxt(path, rows, delimiter=",", header=header, comments="", fmt="%.17g")

    def mean(self) -> dict[str, float]:
        """Posterior mean of each parameter."""
        return {n: float(self.samples[:, j].mean()) for j, n in enumerate(self.names)}


def hmc(
    model: Any,
    num_samples: int = 1000,
    burn_in: int = 200,
    step_size: float = 0.1,
    num_leapfrog: int = 10,
    seed: int = 1,
    include_jacobian: bool = True,
    allow_improper: bool = False,
    fix: Sequence[str] = (),
) -> HmcResult:
    """Sample the posterior over the model's free hyperparameters with HMC.

    The step size is adapted by dual averaging over ``burn_in`` transitions
    (those draws are discarded). Parameters listed in ``fix`` are held fixed for
    the run; the model's parameters are restored afterwards. Raises
    :class:`gpc.GpcError` if the target is improper (add priors with
    :func:`set_prior`, or pass ``allow_improper=True``).
    """
    if not hasattr(model, "hmc"):
        raise TypeError(
            f"{type(model).__name__} does not support HMC. Mini-batch ELBOs "
            "(DeepGP, SVGP, stochastic Bayesian GPLVM) are noisy, which breaks "
            "the accept step; use GPRegression, SparseGPRegression, GPLVM, "
            "BayesianGPLVM or MRD."
        )
    raw = model.hmc(
        num_samples=num_samples,
        burn_in=burn_in,
        step_size=step_size,
        num_leapfrog=num_leapfrog,
        seed=seed,
        include_jacobian=include_jacobian,
        allow_improper=allow_improper,
        fix=list(fix),
    )
    return HmcResult(
        names=list(raw["names"]),
        samples=np.asarray(raw["samples"]),
        log_posterior=np.asarray(raw["log_posterior"]).reshape(-1),
        accept_rate=float(raw["accept_rate"]),
        step_size=float(raw["step_size"]),
    )
