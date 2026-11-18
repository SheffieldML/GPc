"""Paramz-inspired text formatting for ModelDescription (CIP-001E)."""

from __future__ import annotations

from typing import Any


def _constraint_label(row: Any) -> str:
    if getattr(row, "fixed", False):
        return "fixed"
    t = getattr(row, "transform", "") or ""
    if t in ("exp", "negLogLogit"):
        return "+ve"
    if t == "sigmoid":
        return "[0, 1]"
    if t == "linear" or t == "":
        return ""
    return t


def format_model_description(
    desc: Any,
    *,
    name: str | None = None,
    objective: float | None = None,
) -> str:
    """Render a ModelDescription like GPy/paramz ``print(m)``."""
    title = name or (getattr(desc, "title", None) or "model")
    params = list(getattr(desc, "params", []) or [])
    n_params = len(params)
    n_free = sum(1 for p in params if not getattr(p, "fixed", False))

    obj = objective
    if obj is None and getattr(desc, "has_log_likelihood", False):
        # GPy prints the optimisation objective (negative log likelihood).
        obj = -float(desc.log_likelihood)

    lines: list[str] = []
    lines.append(f"Name : {title}")
    if obj is not None:
        lines.append(f"Objective : {obj:.6g}")
    lines.append(f"Number of Parameters : {n_params}")
    lines.append(f"Number of Optimization Parameters : {n_free}")
    lines.append("Updates : True")
    lines.append("Parameters:")

    # Column widths.
    name_w = max([len("GP_regression.")] + [len(p.path) for p in params] + [16])
    val_w = 12
    cons_w = max(
        [len("constraints")]
        + [len(_constraint_label(p)) for p in params]
        + [8]
    )

    header = (
        f"  {'GP_regression.':<{name_w}}  |  {'value':>{val_w}}  |  "
        f"{'constraints':<{cons_w}}"
    )
    lines.append(header)
    for p in params:
        val = float(p.value)
        if abs(val) >= 1e4 or (abs(val) < 1e-3 and val != 0.0):
            val_s = f"{val:.4g}"
        else:
            val_s = f"{val:.6g}"
        cons = _constraint_label(p)
        lines.append(
            f"  {p.path:<{name_w}}  |  {val_s:>{val_w}}  |  {cons:<{cons_w}}"
        )
    return "\n".join(lines) + "\n"
