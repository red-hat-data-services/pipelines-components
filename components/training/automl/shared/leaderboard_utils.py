"""Shared leaderboard HTML-building utilities for AutoML evaluation components.

These helpers are embedded into leaderboard component containers via
``embedded_artifact_path`` and imported with bare module imports inside each
component body (KFP adds the embedded directory to ``sys.path`` at runtime).
"""

import html as _html_module
from pathlib import Path


def _round_metrics(metrics: dict, decimals: int = 4) -> dict:
    """Round numeric values in a metrics dict to the given number of decimals.

    Args:
        metrics: Dictionary of metric names to values.
        decimals: Number of decimal places to round to.

    Returns:
        Dictionary with numeric values rounded; non-numeric values unchanged.
    """
    return {k: round(v, decimals) if isinstance(v, (int, float)) else v for k, v in metrics.items()}


def _build_leaderboard_table(df) -> str:
    """Build table HTML with Notebook and Predictor as separate URI columns.

    Args:
        df: DataFrame with columns including ``notebook``, ``predictor``, and metric columns.
            Must have an ``index.name`` attribute and support ``iterrows()``.

    Returns:
        HTML string for the leaderboard ``<table>``.
    """
    display_cols = [c for c in df.columns if c not in ("notebook", "predictor")]
    rows = []
    rows.append(
        "<thead><tr>"
        + "".join(f"<th>{_html_module.escape(str(c))}</th>" for c in [df.index.name or "rank"] + display_cols)
        + "<th>Notebook</th><th>Predictor</th></tr></thead><tbody>"
    )
    for idx, row in df.iterrows():
        cells = [f"<td>{_html_module.escape(str(idx))}</td>"]
        for col in display_cols:
            val = row[col]
            cells.append(f"<td>{_html_module.escape(str(val))}</td>")
        notebook_uri = _html_module.escape(str(row["notebook"]))
        predictor_uri = _html_module.escape(str(row["predictor"]))
        cells.append(
            f'<td class="uri-cell">'
            f'<a href="{notebook_uri}" class="uri-link" data-uri="{notebook_uri}" target="_blank" rel="noopener">URI</a>'  # noqa: E501
            f'<div class="uri-popover" role="dialog" aria-label="URI" hidden>'
            f'<pre class="uri-popover-text"></pre>'
            f'<button type="button" class="uri-popover-close" aria-label="Close">×</button>'
            f"</div></td>"
        )
        cells.append(
            f'<td class="uri-cell">'
            f'<a href="{predictor_uri}" class="uri-link" data-uri="{predictor_uri}" target="_blank" rel="noopener">URI</a>'  # noqa: E501
            f'<div class="uri-popover" role="dialog" aria-label="URI" hidden>'
            f'<pre class="uri-popover-text"></pre>'
            f'<button type="button" class="uri-popover-close" aria-label="Close">×</button>'
            f"</div></td>"
        )
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return "<table>" + "".join(rows) + "</tbody></table>"


def _build_leaderboard_html(
    template_path: str | Path,
    table_html: str,
    eval_metric: str,
    best_model_name: str,
    num_models: int,
) -> str:
    """Build leaderboard HTML by substituting placeholders in the shared template.

    Args:
        template_path: Path to ``leaderboard_html_template.html``.
        table_html: HTML string produced by ``_build_leaderboard_table``.
        eval_metric: Metric name displayed in the header.
        best_model_name: Display name of the top-ranked model.
        num_models: Total number of models in the leaderboard.

    Returns:
        Complete HTML string for the leaderboard page.
    """
    with Path(template_path).open("r", encoding="utf-8") as f:
        template = f.read()
    return (
        template.replace("__TABLE_HTML__", table_html)
        .replace("__NUM_MODELS__", str(num_models))
        .replace("__EVAL_METRIC__", eval_metric)
        .replace("__BEST_MODEL_NAME__", best_model_name)
    )
