"""Read/write adapter for the GP model interchange format (CIP-0016, gp-scuderia).

Wire documents are plain dicts (see ``gp-scuderia/schemas/model.schema.json``);
file I/O is the caller's job (``gp_scuderia.load`` / ``dump``), so this module
has no dependency beyond numpy. Wire parameters are natural and positive; the
unit conversions to GPc storage happen here, off the optimiser path.

Conventions
-----------
* ``lengthscale`` scalar -> ``RBF`` (GPc stores ``inverseWidth``; the façade
  converts, CIP-0014).
* ``lengthscale`` list (ARD) -> ``rbfard``: GPc stores one ``inverseWidth``
  (``lengthscale``) and per-dimension ``inputScale_j`` in [0, 1] with
  ``1/l_j**2 = inverseWidth * inputScale_j`` (CRbfardKern::computeElement).
  We write the canonical split ``inputScale_j = (min(l)/l_j)**2`` (largest
  scale is 1); reading inverts it exactly.
* Gaussian noise: ``likelihood.variance`` (FTC) or ``beta = 1/variance`` (sparse).
"""
from __future__ import annotations

import numpy as np

from . import kern as K
from . import models as M

SCHEMA_VERSION = "0.1"

_KERN_CLS = {
    "kern.rbf": K.RBF,
    "kern.matern32": K.Matern32,
    "kern.white": K.White,
    "kern.bias": K.Bias,
}


class InterchangeError(ValueError):
    """The document cannot be represented faithfully by GPc."""


def _leaves(node):
    if node["type"] in ("kern.add",):
        for p in node["parts"]:
            yield from _leaves(p)
    elif node["type"] == "kern.prod":
        raise InterchangeError("kern.prod is not yet supported by the GPc adapter")
    else:
        yield node


def _kernel_from_wire(node, input_dim):
    ard_scales = {}  # leaf order index -> per-dimension scales
    parts = []
    for i, leaf in enumerate(_leaves(node)):
        t = leaf["type"]
        if t not in _KERN_CLS:
            raise InterchangeError("GPc adapter: unsupported kernel type %r" % t)
        if t in ("kern.rbf", "kern.matern32"):
            ls = leaf["lengthscale"]
            if isinstance(ls, (list, tuple)):
                if t != "kern.rbf":
                    raise InterchangeError("ARD only supported for kern.rbf")
                ls = np.asarray(ls, float)
                if len(ls) != input_dim:
                    raise InterchangeError("ARD lengthscale length != input_dim")
                base = float(ls.min())
                parts.append(K.RBF(input_dim, variance=leaf["variance"],
                                   lengthscale=base, ARD=True))
                ard_scales[len(parts) - 1] = (base / ls) ** 2
                continue
            parts.append(_KERN_CLS[t](input_dim, variance=leaf["variance"],
                                      lengthscale=float(ls)))
        else:
            parts.append(_KERN_CLS[t](input_dim, variance=leaf["variance"]))
    kern = parts[0]
    for p in parts[1:]:
        kern = kern + p
    return kern, ard_scales


def from_wire(doc):
    """Build a GPc model from a wire document (must embed ``data``)."""
    if doc.get("schema_version") != SCHEMA_VERSION:
        raise InterchangeError("unsupported schema_version %r" % doc.get("schema_version"))
    d = int(doc["input_dim"])
    if "data" not in doc:
        raise InterchangeError("GPc adapter needs embedded data")
    X = np.asarray(doc["data"]["X"], float)
    Y = np.asarray(doc["data"]["Y"], float)
    if X.shape[1] != d:
        raise InterchangeError("data.X width %d != input_dim %d" % (X.shape[1], d))
    kern, ard = _kernel_from_wire(doc["kernel"], d)
    var = float(doc["likelihood"]["variance"])
    approx = doc.get("approximation", "ftc")
    if approx == "ftc":
        model = M.GPRegression(X, Y, kern)
        model.set_by_path("likelihood.variance", var)
    else:
        Z = np.asarray(doc["inducing_inputs"], float)
        model = M.SparseGPRegression(
            X, Y, Z, kernel=kern, approx=approx,
            sparse_linalg=doc.get("sparse_linalg", "lq"))
        model.set_by_path("likelihood.variance", var)
    for idx, scales in ard.items():
        label = "kern.rbf"
        for j, s in enumerate(scales):
            model.set_by_path("%s.inputScale_%d" % (label, j), float(s))
    return model


def _get(model, path):
    return float(model.get_by_path(path))


def to_wire(model, doc_template):
    """Return a wire document describing ``model``.

    ``doc_template`` supplies structure GPc cannot report (kernel tree, data,
    inducing inputs); parameters are read back from the model, so this checks
    the round trip rather than copying the template.
    """
    doc = {k: v for k, v in doc_template.items() if k != "x-reference"}
    paths = set(model.param_paths())

    def leaf_out(leaf):
        out = dict(leaf)
        t = leaf["type"]
        if t == "kern.rbf":
            out["variance"] = _get(model, "kern.rbf.variance")
            base = _get(model, "kern.rbf.lengthscale")
            if isinstance(leaf["lengthscale"], list):
                n = len(leaf["lengthscale"])
                out["lengthscale"] = [
                    float(base / np.sqrt(_get(model, "kern.rbf.inputScale_%d" % j)))
                    for j in range(n)]
            else:
                out["lengthscale"] = base
        elif t in ("kern.white", "kern.bias"):
            out["variance"] = _get(model, "kern.%s.variance" % t.split(".")[1])
        else:
            raise InterchangeError("unsupported kernel type %r" % t)
        return out

    def walk(node):
        if node["type"] == "kern.add":
            return {**node, "parts": [walk(p) for p in node["parts"]]}
        return leaf_out(node)

    doc["kernel"] = walk(doc_template["kernel"])
    doc["likelihood"] = {**doc_template["likelihood"],
                         "variance": _get(model, "likelihood.variance")}
    return doc


def reference(model, Xnew):
    mean, var = model.predict(np.asarray(Xnew, float))
    return {"log_likelihood": -float(model.objective()),
            "mean": np.asarray(mean).ravel().tolist(),
            "var": np.asarray(var).ravel().tolist()}
