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
    opt_override = getattr(desc, "opt_num_params", None)
    if opt_override is not None and int(opt_override) >= 0:
        n_free = int(opt_override)

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


def format_model_description_html(
    desc: Any,
    *,
    name: str | None = None,
    objective: float | None = None,
) -> str:
    """HTML twin of :func:`format_model_description` for notebook widgets."""
    text = format_model_description(desc, name=name, objective=objective)
    return (
        "<pre style='line-height:1.35;margin:0;font-family:Menlo,Consolas,"
        "monospace;font-size:12px'>"
        f"{html.escape(text)}</pre>"
    )


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


def _format_runtime(seconds: float) -> str:
    if seconds < 60:
        ms = (seconds % 1) * 100
        return f"{int(seconds):02d}s{int(ms):02d}"
    m, s = divmod(seconds, 60)
    if m > 59:
        h, m = divmod(m, 60)
        if h > 23:
            d, h = divmod(h, 24)
            return f"{int(d):02d}d{int(h):02d}h{int(m):02d}"
        return f"{int(h):02d}h{int(m):02d}m{int(s):02d}"
    ms = (seconds % 1) * 100
    return f"{int(m):02d}m{int(s):02d}s{int(ms):02d}"


class OptimiserTableProgress:
    """Refresh the paramz-style model table in place during ``optimize``.

    Prefer paramz's notebook path: ``ipywidgets`` progress + status HTML +
    live model HTML. Fall back to a single ``display_id`` cell in ZMQ Jupyter
    without widgets, then ANSI cursor-up rewrite in a plain terminal.
    """

    def __init__(
        self,
        model: Any,
        *,
        name: str = "GP regression",
        max_iters: int = 100,
    ) -> None:
        self.model = model
        self.name = name
        self.max_iters = max(1, int(max_iters))
        self._start = time.time()
        self._last_draw = 0.0
        self._lines = 0
        self._handle: Any | None = None
        self._finished = False
        self._mode = "ansi"
        self._progress: Any | None = None
        self._status_html: Any | None = None
        self._model_html: Any | None = None
        self._box: Any | None = None
        self._setup_backend()

    def _setup_backend(self) -> None:
        # Match paramz: try widgets under any IPython first.
        try:
            from IPython import get_ipython
            from IPython.display import display
            from ipywidgets import HTML, HBox, IntProgress, VBox

            if get_ipython() is None:
                raise RuntimeError("not under IPython")

            self._status_html = HTML(value="")
            self._progress = IntProgress(min=0, max=self.max_iters, value=0)
            self._model_html = HTML(value="")
            left = VBox(children=[self._progress, self._status_html])
            self._box = HBox(children=[left, self._model_html])
            display(self._box)
            self._mode = "widgets"
            return
        except Exception:
            pass

        if _jupyter_zmq_shell() is not None:
            self._mode = "display_id"
            return

        self._mode = "ansi"

    def __call__(self, event: Any) -> None:
        if self._finished:
            return

        phase = getattr(event, "phase", None)
        opt = _OPT_NAMES.get(int(getattr(event, "optimiser", 1)), "optimiser")
        it = int(getattr(event, "iter", 0))
        obj = float(getattr(event, "objective", float("nan")))
        scale = float(getattr(event, "step_or_scale", float("nan")))
        grad = float(getattr(event, "grad_norm", float("nan")))
        elapsed = time.time() - self._start

        phase_name = getattr(phase, "name", str(phase))
        status = "running"
        footer = ""
        if phase_name == "MaxIters":
            status = "maxiters"
            footer = "Warning: Maximum number of iterations has been exceeded\n"
        elif phase_name == "Converged":
            status = "converged"
            footer = "Optimization converged.\n"
        elif phase_name == "Failed":
            status = "failed"
            msg = getattr(event, "message", "") or "Optimization failed."
            footer = f"{msg}\n"
        elif phase_name == "Start":
            status = "start"

        # Throttle mid-run redraws like paramz: always in the first 0.2s,
        # then ~1 Hz; always show start/terminal frames.
        now = time.time()
        terminal = phase_name in ("Start", "Converged", "MaxIters", "Failed")
        if (
            not terminal
            and elapsed >= 0.2
            and (now - self._last_draw) < 1.0
        ):
            return
        self._last_draw = now

        obj_for_table = obj if math.isfinite(obj) else None
        table = format_model_description(
            self.model.describe(),
            name=self.name,
            objective=obj_for_table,
        )
        table_html = format_model_description_html(
            self.model.describe(),
            name=self.name,
            objective=obj_for_table,
        )

        if self._mode == "widgets":
            self._render_widgets(
                opt=opt,
                it=it,
                obj=obj,
                grad=grad,
                elapsed=elapsed,
                status=status,
                table_html=table_html,
                terminal=terminal,
            )
        else:
            status_bits = [f"Running {opt}", f"iter {it}", f"{elapsed:.2f}s"]
            if math.isfinite(obj):
                status_bits.append(f"objective {obj:.6g}")
            if math.isfinite(scale):
                status_bits.append(f"scale {scale:.6g}")
            block = f"{' · '.join(status_bits)}\n{table}{footer}"
            self._render_text(block)

        if terminal and phase_name != "Start":
            self._finished = True
            if self._mode == "ansi" and self._lines:
                sys.stdout.write("\n")
                sys.stdout.flush()
                self._lines = 0

    def _render_widgets(
        self,
        *,
        opt: str,
        it: int,
        obj: float,
        grad: float,
        elapsed: float,
        status: str,
        table_html: str,
        terminal: bool,
    ) -> None:
        assert self._progress is not None
        assert self._status_html is not None
        assert self._model_html is not None

        rows = [
            ("optimizer", opt),
            ("runtime", _format_runtime(elapsed)),
            ("iteration", f"{it}/{self.max_iters}"),
            (
                "objective",
                f"{obj: .6E}" if math.isfinite(obj) else "n/a",
            ),
            (
                "||gradient||",
                f"{grad: .6E}" if math.isfinite(grad) else "n/a",
            ),
            ("status", status),
        ]
        body = "".join(
            "<tr>"
            f"<td style='padding:2px 6px;font-weight:bold'>{html.escape(k)}</td>"
            f"<td style='padding:2px 6px;font-family:monospace'>{html.escape(v)}</td>"
            "</tr>"
            for k, v in rows
        )
        self._status_html.value = (
            "<table style='border-collapse:collapse;font-size:12px'>"
            f"{body}</table>"
        )
        self._progress.value = min(it + (1 if terminal else 0), self.max_iters)
        if status == "converged":
            self._progress.bar_style = "success"
        elif status == "maxiters":
            self._progress.bar_style = "warning"
        elif status == "failed":
            self._progress.bar_style = "danger"
        self._model_html.value = table_html

    def _render_text(self, block: str) -> None:
        if self._mode == "display_id":
            try:
                from IPython.display import HTML, display
            except ImportError:
                self._mode = "ansi"
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
