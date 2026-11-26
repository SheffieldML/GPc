"""Read/write adapter for the GP model interchange format (CIP-0016, gp-scuderia).

The mapping logic lives once, in C++ (``CInterchange``): lengthscale and ARD
conversion, noise semantics (kernel White and ``likelihood.variance`` both add),
zero mean, and the refusal rules. This module only moves documents across the
boundary as JSON, so it has no dependency beyond the standard library.

Wire documents are plain dicts (see ``gp-scuderia/schemas/model.schema.json``);
file I/O is the caller's job (``gp_scuderia.load`` / ``dump``).
"""
from __future__ import annotations

import json

from . import _gpc

SCHEMA_VERSION = "0.1"


class InterchangeError(ValueError):
    """The document cannot be represented faithfully by GPc."""


def from_wire(doc):
    """Build a GPc model from a wire document (must embed ``data``).

    Returns an ``InterchangeModel`` with ``objective``, ``predict``,
    ``set_by_path``/``get_by_path``, ``set_allow_white_and_noise`` and
    ``to_json``.
    """
    try:
        return _gpc.InterchangeModel(json.dumps(doc))
    except _gpc.GpcError as e:
        raise InterchangeError(str(e)) from e


def to_wire(model, doc_template=None):
    """Return the wire document describing ``model``.

    Structure and ``x-*`` fields come from the document the model was built
    from; parameters are read back from the model. ``doc_template`` is accepted
    for backwards compatibility and ignored.
    """
    return json.loads(model.to_json())


def reference(model, Xnew):
    mean, var = model.predict(Xnew)
    out = {"log_likelihood": -float(model.objective()),
           "mean": mean.ravel().tolist(),
           "var": var.ravel().tolist()}
    if model.is_laplace():
        mu, s2 = model.predict_latent(Xnew)
        out["latent_mean"] = mu.ravel().tolist()
        out["latent_var"] = s2.ravel().tolist()
    return out
