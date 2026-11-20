"""Paramz-inspired text formatting and in-place optimiser progress (CIP-001E)."""

from __future__ import annotations

import html
import math
import sys
import time
from typing import Any

_OPT_NAMES = {
    0: "cg",
    1: "scg",
    2: "gd",
    3: "bfgs",
    4: "lbfgs",
}


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


def _jupyter_zmq_shell() -> Any | None:
    """Return the IPython shell only when running under Jupyter (ZMQ).

    Terminal IPython also defines ``get_ipython()``, but ``clear_output`` does
    not rewrite prior prints there — tables stack. Use ANSI in that case.
    """
    try:
        from IPython import get_ipython
    except ImportError:
        return None
    ip = get_ipython()
    if ip is None:
        return None
    if ip.__class__.__name__ == "ZMQInteractiveShell":
        return ip
    return None


class OptimiserTableProgress:
    """Refresh the paramz-style model table in place during ``optimize``.

    Jupyter (ZMQ): update a single ``display_id`` HTML ``<pre>`` cell.
    Terminal / terminal IPython: ANSI cursor-up rewrite of the same block.
    """

    def __init__(self, model: Any, *, name: str = "GP regression") -> None:
        self.model = model
        self.name = name
        self._start = time.time()
        self._lines = 0
        self._jupyter = _jupyter_zmq_shell() is not None
        self._handle: Any | None = None
        self._finished = False

    def __call__(self, event: Any) -> None:
        if self._finished:
            return

        phase = getattr(event, "phase", None)
        opt = _OPT_NAMES.get(int(getattr(event, "optimiser", 1)), "optimiser")
        it = int(getattr(event, "iter", 0))
        obj = float(getattr(event, "objective", float("nan")))
        scale = float(getattr(event, "step_or_scale", float("nan")))
        elapsed = time.time() - self._start

        obj_for_table = obj if math.isfinite(obj) else None
        table = format_model_description(
            self.model.describe(),
            name=self.name,
            objective=obj_for_table,
        )

        status_bits = [f"Running {opt}", f"iter {it}", f"{elapsed:.2f}s"]
        if math.isfinite(obj):
            status_bits.append(f"objective {obj:.6g}")
        if math.isfinite(scale):
            status_bits.append(f"scale {scale:.6g}")
        status = " · ".join(status_bits)

        footer = ""
        phase_name = getattr(phase, "name", str(phase))
        if phase_name == "MaxIters":
            footer = "Warning: Maximum number of iterations has been exceeded\n"
        elif phase_name == "Converged":
            footer = "Optimization converged.\n"
        elif phase_name == "Failed":
            msg = getattr(event, "message", "") or "Optimization failed."
            footer = f"{msg}\n"

        block = f"{status}\n{table}{footer}"
        self._render(block)

        if phase_name in ("Converged", "MaxIters", "Failed"):
            self._finished = True
            if not self._jupyter and self._lines:
                sys.stdout.write("\n")
                sys.stdout.flush()
                self._lines = 0

    def _render(self, block: str) -> None:
        if self._jupyter:
            try:
                from IPython.display import HTML, display
            except ImportError:
                self._jupyter = False
            else:
                payload = HTML(
                    "<pre style='line-height:1.3;margin:0'>"
                    f"{html.escape(block)}</pre>"
                )
                if self._handle is None:
                    self._handle = display(payload, display_id=True)
                else:
                    self._handle.update(payload)
                return

        if self._lines > 0:
            sys.stdout.write(f"\033[{self._lines}A\r")
        sys.stdout.write(block)
        sys.stdout.write("\033[J")
        sys.stdout.flush()
        self._lines = block.count("\n")
