"""Standalone HTML report and the compact notebook view of a result."""

from __future__ import annotations

import base64
import io
from html import escape
from importlib.resources import files
from string import Template
from typing import TYPE_CHECKING

import numpy as np

from mchermes import __version__
from mchermes.report import ROWS, fmt

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from mchermes.result import Result


def _asset(name: str) -> str:
    return files("mchermes").joinpath("templates").joinpath(name).read_text(encoding="utf-8")


def _svg(fig: Figure) -> str:
    import matplotlib.pyplot as plt

    buffer = io.StringIO()
    fig.savefig(buffer, format="svg", bbox_inches="tight")
    plt.close(fig)
    return buffer.getvalue()


def _chart(draw: object, label: str) -> str:
    """Light and dark renderings of one chart; CSS shows the one matching the theme."""
    import matplotlib.pyplot as plt

    images = []
    with plt.rc_context({"svg.fonttype": "none"}):
        for theme in ("light", "dark"):
            svg = _svg(draw(theme))  # type: ignore[operator]
            data = base64.b64encode(svg.encode()).decode()
            images.append(
                f'<img class="chart chart--{theme}" alt="{escape(label)}" '
                f'src="data:image/svg+xml;base64,{data}">'
            )
    return "".join(images)


def _cell(value: float, is_probability: bool) -> str:
    text = f"{value:.1%}" if is_probability else fmt(value)
    css = ' class="negative"' if value < 0 and not is_probability else ""
    return f"<td{css}>{escape(text)}</td>"


def metrics_table(result: Result) -> str:
    """Metrics with one column per option; the recommended option is highlighted."""
    best = result.best_option()
    head = "".join(
        f'<th class="best">★ {escape(o)}</th>' if o == best else f"<th>{escape(o)}</th>"
        for o in result.options
    )
    body = []
    for label, attribute, is_probability in ROWS:
        cells = "".join(
            _cell(getattr(result.stats[o], attribute), is_probability) for o in result.options
        )
        body.append(f"<tr><th scope=row>{escape(label)}</th>{cells}</tr>")
    return (
        f"<table><thead><tr><th>Metric</th>{head}</tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table>"
    )


def _voi_table(result: Result) -> str:
    rows = result.value_of_information()
    if not rows:
        return '<p class="empty">No uncertain inputs.</p>'
    top = max(r.value for r in rows) or 1.0
    body = "".join(
        f"<tr><td><code>{escape(r.variable)}</code></td><td>{escape(fmt(r.value))}</td>"
        f"<td>{r.share_of_evpi:.0%}</td><td class=bar-cell>"
        f'<span class="bar" style="width:{100 * r.value / top:.1f}%"></span></td></tr>'
        for r in rows
    )
    return (
        "<table><thead><tr><th>Input</th><th>EVPPI</th><th>Share of EVPI</th><th></th></tr>"
        f"</thead><tbody>{body}</tbody></table>"
    )


def _inputs_table(result: Result) -> str:
    body = []
    for name, values in result.inputs.items():
        p5, p50, p95 = np.percentile(values, [5, 50, 95])
        body.append(
            f"<tr><td><code>{escape(name)}</code></td><td>{escape(fmt(float(values.mean())))}</td>"
            f"<td>{escape(fmt(float(p5)))}</td><td>{escape(fmt(float(p50)))}</td>"
            f"<td>{escape(fmt(float(p95)))}</td></tr>"
        )
    if not body:
        return '<p class="empty">No uncertain inputs.</p>'
    return (
        "<table><thead><tr><th>Input</th><th>Mean</th><th>P5</th><th>Median</th><th>P95</th>"
        f"</tr></thead><tbody>{''.join(body)}</tbody></table>"
    )


def _tornado_tabs(result: Result) -> str:
    from mchermes.plotting import plot_tornado

    options = [o for o in result.options if result.stats[o].std > 0]
    if not options:
        return '<p class="empty">Every option is certain: no inputs move the outcome.</p>'
    tabs, panels = [], []
    for i, option in enumerate(options):
        selected = "true" if i == 0 else "false"
        tabs.append(
            f'<button class="tab" type="button" role="tab" id="tab-{i}" aria-selected="{selected}" '
            f'aria-controls="panel-{i}" tabindex="{0 if i == 0 else -1}">{escape(option)}</button>'
        )
        chart = _chart(
            lambda theme, o=option: plot_tornado(result, o, theme=theme, header=False),
            f"Drivers of {option}",
        )
        hidden = "" if i == 0 else " hidden"
        panels.append(
            f'<div class="tab-panel" role="tabpanel" id="panel-{i}" '
            f'aria-labelledby="tab-{i}"{hidden}>{chart}</div>'
        )
    tablist = f'<div class="tabs" role="tablist">{"".join(tabs)}</div>' if len(tabs) > 1 else ""
    return tablist + "".join(panels)


def render_report(result: Result, *, title: str | None = None) -> str:
    """The full standalone report page (no external requests)."""
    from mchermes.plotting import plot_cdf, plot_distributions

    best = result.best_option()
    stats = result.stats[best]
    tiles = {
        "Expected outcome": fmt(stats.mean),
        "P(loss)": f"{stats.prob_loss:.1%}",
        "P(best)": f"{stats.prob_best:.1%}",
        "Worst 5% average": fmt(stats.cvar_95),
        "Value of perfect info": fmt(result.evpi()),
    }
    page = Template(_asset("report.html")).substitute(
        version=__version__,
        title=escape(title or result.name),
        meta=escape(
            f"{result.n:,} simulated futures · seed {result.seed} · criterion {result.criterion}"
        ),
        best=escape(best),
        why=escape(
            f"Best by {result.criterion.replace('_', ' ')}; it wins in "
            f"{stats.prob_best:.0%} of simulated futures."
        ),
        tiles="\n".join(
            f'      <div class="tile"><dt>{escape(k)}</dt><dd>{escape(v)}</dd></div>'
            for k, v in tiles.items()
        ),
        chart_distributions=_chart(
            lambda theme: plot_distributions(result, theme=theme, header=False),
            "Outcome distributions",
        ),
        chart_cdf=_chart(
            lambda theme: plot_cdf(result, theme=theme, header=False), "Cumulative outcome"
        ),
        tornado_tabs=_tornado_tabs(result),
        metrics_table=metrics_table(result),
        voi_table=_voi_table(result),
        inputs_table=_inputs_table(result),
        evpi=escape(fmt(result.evpi())),
        seed=result.seed,
        css=_asset("report.css"),
        js=_asset("report.js"),
    )
    return page


def summary_html(result: Result) -> str:
    """Compact table for Jupyter (``Result._repr_html_``)."""
    best = result.best_option()
    stats = result.stats[best]
    return (
        f"<div><strong>{escape(result.name)}</strong> "
        f"<span style='opacity:.7'>· {result.n:,} samples · seed {result.seed}</span>"
        f"{metrics_table(result)}"
        f"<p>Recommendation: <strong>{escape(best)}</strong> (expected {escape(fmt(stats.mean))}, "
        f"P(loss) {stats.prob_loss:.1%}, best in {stats.prob_best:.1%} of futures)</p></div>"
    )


__all__ = ["metrics_table", "render_report", "summary_html"]
