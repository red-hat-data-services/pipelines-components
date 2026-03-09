from kfp import dsl


@dsl.component(
    base_image="registry.redhat.io/rhoai/odh-pipeline-runtime-datascience-cpu-py312-rhel9@sha256:f9844dc150592a9f196283b3645dda92bd80dfdb3d467fa8725b10267ea5bdbc",
)
def leaderboard_evaluation(
    rag_patterns: dsl.InputPath(dsl.Artifact),
    html_artifact: dsl.Output[dsl.HTML],
    optimization_metric: str = "faithfulness",
):
    """Build an HTML leaderboard artifact from RAG pattern evaluation results.

    Reads pattern.json from each subdirectory of rag_patterns (produced by
    rag_templates_optimization) and generates a single HTML table: Pattern_Name,
    mean_* metrics (e.g. mean_answer_correctness, mean_faithfulness), then
    config columns (chunking.*, embeddings.model_id, retrieval.*,
    generation.model_id). Writes the HTML to html_artifact.path.

    Args:
        rag_patterns: Path to the directory of RAG patterns; each subdir contains
            pattern.json (pattern_name, indexing_params, rag_params, scores,
            execution_time, final_score).
        html_artifact: Output HTML artifact; the leaderboard table is written to
            html_artifact.path (single file).
        optimization_metric: Name of the metric used to rank patterns (e.g. faithfulness,
            answer_correctness, context_correctness). Shown in the leaderboard
            subtitle. Defaults to "faithfulness".
    """
    import html
    import json
    from pathlib import Path

    # Fallback keys when ai4rag uses different structure (embedding vs embeddings, flat keys)
    _config_column_fallbacks = {
        "embeddings.model_id": ("embedding_model", "embedding.model_id"),
        "generation.model_id": ("foundation_model",),
    }

    def _get_nested(params: dict, key: str):
        """Resolve dotted key from flat or nested dict (e.g. chunking.method)."""
        if not params:
            return None
        if key in params:
            return params[key]
        parts = key.split(".", 1)
        if len(parts) == 2:
            outer = params.get(parts[0])
            if isinstance(outer, dict):
                return outer.get(parts[1])
        return None

    def _get_config_value(merged: dict, col: str):
        """Get config column value, with fallbacks for ai4rag structure (embedding vs embeddings, flat keys)."""
        val = _get_nested(merged, col)
        if val is not None:
            return val
        fallbacks = _config_column_fallbacks.get(col)
        if not fallbacks:
            return None
        for fallback in fallbacks:
            if "." in fallback:
                val = _get_nested(merged, fallback)
            else:
                val = merged.get(fallback) if merged else None
            if val is not None:
                return val
        return None

    def _merge_params(indexing_params: dict, rag_params: dict) -> dict:
        merged = dict(indexing_params or {})
        merged.update(rag_params or {})
        return merged

    def _settings_from_rag_pattern(e: dict) -> dict | None:
        """Build leaderboard config dict from rag_pattern.settings (legacy nested schema)."""
        rp = (e.get("rag_pattern") or {}).get("settings")
        if not rp:
            return None
        return {
            "chunking": rp.get("chunking") or {},
            "embeddings": {"model_id": (rp.get("embedding") or {}).get("model_id")},
            "retrieval": {
                "method": rp.get("method"),
                "number_of_chunks": rp.get("number_of_chunks"),
            },
            "generation": {"model_id": (rp.get("generation") or {}).get("model_id")},
        }

    def _normalize_flat_settings(settings: dict | None) -> dict | None:
        """Build leaderboard config dict from flat pattern.json settings (embedding, retrieval)."""
        if not settings:
            return None
        emb = settings.get("embedding") or settings.get("embeddings") or {}
        gen = settings.get("generation") or {}
        # Support nested (embedding.model_id) or flat (embedding_model from ai4rag) keys
        emb_model_id = emb.get("model_id") if isinstance(emb, dict) else None
        if emb_model_id is None and isinstance(settings.get("embedding_model"), str):
            emb_model_id = settings.get("embedding_model")
        gen_model_id = gen.get("model_id") if isinstance(gen, dict) else None
        if gen_model_id is None and isinstance(settings.get("foundation_model"), str):
            gen_model_id = settings.get("foundation_model")
        return {
            "chunking": settings.get("chunking") or {},
            "embeddings": {"model_id": emb_model_id},
            "retrieval": settings.get("retrieval") or {},
            "generation": {"model_id": gen_model_id},
        }

    def _metric_to_mean_key(metric: str) -> str:
        return "mean_" + metric

    def _metric_display_name(metric: str) -> str:
        """Format metric name for display (e.g. answer_correctness -> answer correctness)."""
        if not metric:
            return "optimization metric"
        return metric.replace("_", " ").strip()

    def _build_leaderboard_html(
        header_row: str,
        table_body: str,
        best_pattern_name: str,
        num_patterns: int,
        eval_metric: str,
    ) -> str:
        """Build a styled HTML document for the RAG leaderboard (aligned with AutoML leaderboard style)."""
        metric_label = html.escape(_metric_display_name(eval_metric))
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RAG Patterns Leaderboard</title>
  <style>
    :root {{
      --bg: #0f1419;
      --surface: #1a2332;
      --surface-hover: #243044;
      --border: #2d3a4f;
      --text: #e6edf3;
      --text-muted: #8b949e;
      --accent: #58a6ff;
      --accent-dim: #388bfd66;
      --gold: #f0b429;
      --silver: #a8b2c1;
      --bronze: #c9a227;
      --success: #3fb950;
      --radius: 12px;
      --font: 'Segoe UI', system-ui, -apple-system, sans-serif;
    }}
    * {{
      box-sizing: border-box;
    }}
    body {{
      margin: 0;
      padding: 2rem;
      font-family: var(--font);
      background: var(--bg);
      color: var(--text);
      line-height: 1.5;
      min-height: 100vh;
    }}
    .container {{
      margin: 0 auto;
      width: 100%;
      max-width: 960px;
    }}
    @media (min-width: 1200px) {{
      .container {{ max-width: 1100px; }}
    }}
    @media (min-width: 1400px) {{
      .container {{ max-width: 1300px; }}
    }}
    @media (min-width: 1600px) {{
      .container {{ max-width: 1500px; }}
    }}
    @media (min-width: 1920px) {{
      .container {{ max-width: min(1800px, 92vw); }}
    }}
    header {{
      margin-bottom: 2rem;
      padding-bottom: 1.5rem;
      border-bottom: 1px solid var(--border);
    }}
    h1 {{
      margin: 0 0 0.25rem 0;
      font-size: 1.75rem;
      font-weight: 600;
      letter-spacing: -0.02em;
    }}
    .subtitle {{
      color: var(--text-muted);
      font-size: 0.9rem;
    }}
    .badge {{
      display: inline-block;
      margin-left: 0.5rem;
      padding: 0.2rem 0.5rem;
      font-size: 0.75rem;
      font-weight: 600;
      border-radius: 6px;
      background: var(--accent-dim);
      color: var(--accent);
    }}
    .card {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      overflow: hidden;
      box-shadow: 0 4px 24px rgba(0,0,0,0.25);
    }}
    .leaderboard-wrap {{
      overflow-x: auto;
    }}
    .leaderboard-wrap table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.9rem;
    }}
    .leaderboard-wrap th {{
      text-align: left;
      padding: 1rem 1.25rem;
      background: var(--surface-hover);
      color: var(--text-muted);
      font-weight: 600;
      font-size: 0.75rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      border-bottom: 1px solid var(--border);
    }}
    .leaderboard-wrap td {{
      padding: 1rem 1.25rem;
      border-bottom: 1px solid var(--border);
    }}
    .leaderboard-wrap tr:last-child td {{
      border-bottom: none;
    }}
    .leaderboard-wrap tbody tr:hover {{
      background: var(--surface-hover);
    }}
    .leaderboard-wrap tbody tr.rank-1 {{
      background: linear-gradient(90deg, rgba(240,180,41,0.12) 0%, transparent 100%);
    }}
    .leaderboard-wrap tbody tr.rank-1 td:first-child {{
      color: var(--gold);
      font-weight: 700;
    }}
    .leaderboard-wrap .metric-cell {{
      font-variant-numeric: tabular-nums;
      color: var(--success);
    }}
    .best-model-footer {{
      margin-top: 1.5rem;
      padding: 1rem 1.25rem;
      background: var(--surface-hover);
      border-radius: 8px;
      font-size: 0.9rem;
      color: var(--text-muted);
    }}
    .best-model-footer strong {{
      color: var(--gold);
    }}
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1>RAG Patterns Leaderboard</h1>
      <p class="subtitle">
        Ranked by <span class="badge">{metric_label}</span> (best first) · {num_patterns} pattern(s)
      </p>
    </header>
    <div class="card">
      <div class="leaderboard-wrap">
        <table>
          <thead>
            <tr>{header_row}</tr>
          </thead>
          <tbody>
            {table_body}
          </tbody>
        </table>
      </div>
    </div>
    <div class="best-model-footer">
      Best pattern: <strong>{html.escape(best_pattern_name)}</strong>
    </div>
  </div>
</body>
</html>"""

    rag_patterns_dir = Path(rag_patterns)
    if not rag_patterns_dir.is_dir():
        raise FileNotFoundError("rag_patterns path is not a directory: %s" % rag_patterns_dir)

    evaluations = []
    for subdir in sorted(rag_patterns_dir.iterdir()):
        if not subdir.is_dir():
            continue
        pattern_file = subdir / "pattern.json"
        if not pattern_file.is_file():
            continue
        with pattern_file.open("r", encoding="utf-8") as f:
            evaluations.append(json.load(f))

    # Sort by optimization metric score descending (highest first); missing scores last
    def _optimization_score(e):
        v = e.get("final_score")
        if v is not None:
            try:
                return (False, -float(v))
            except (TypeError, ValueError):
                pass
        raw = e.get("scores") or {}
        aggregate = raw.get("scores") if isinstance(raw.get("scores"), dict) else raw
        for _k, info in (aggregate or {}).items():
            if isinstance(info, dict):
                mean = info.get("mean")
                if mean is not None:
                    try:
                        return (False, -float(mean))
                    except (TypeError, ValueError):
                        pass
        return (True, 0)

    evaluations.sort(key=_optimization_score)

    # Default column order: metrics first, then RAG config (chunking, embeddings, retrieval, generation).
    leaderboard_metric_columns = [
        "mean_answer_correctness",
        "mean_faithfulness",
        "mean_context_correctness",
    ]
    leaderboard_config_columns = [
        "chunking.method",
        "chunking.chunk_size",
        "chunking.chunk_overlap",
        "embeddings.model_id",
        "retrieval.method",
        "retrieval.number_of_chunks",
        "generation.model_id",
    ]
    # Build metric columns present in data (preferred order above). Support flat scores
    # (metric dict at top level) or nested scores.scores.
    all_metric_names = []
    for e in evaluations:
        raw = e.get("scores") or {}
        aggregate = raw.get("scores") if isinstance(raw.get("scores"), dict) else raw
        for m in aggregate or {}:
            if m not in all_metric_names:
                all_metric_names.append(m)
    metric_columns = [c for c in leaderboard_metric_columns if c.replace("mean_", "", 1) in all_metric_names]
    for m in all_metric_names:
        col = _metric_to_mean_key(m)
        if col not in metric_columns:
            metric_columns.append(col)

    config_columns = list(leaderboard_config_columns)
    headers = ["Pattern_Name"] + metric_columns + config_columns
    header_row = "".join("<th>%s</th>" % html.escape(h) for h in headers)

    rows = []
    for i, e in enumerate(evaluations):
        pattern_name = e.get("name") or e.get("pattern_name") or (e.get("rag_pattern") or {}).get("name", "—")
        raw = e.get("scores") or {}
        scores = raw.get("scores") if isinstance(raw.get("scores"), dict) else raw
        merged = (
            _settings_from_rag_pattern(e)
            or _normalize_flat_settings(e.get("settings"))
            or _merge_params(e.get("indexing_params") or {}, e.get("rag_params") or {})
        )

        cells = [html.escape(str(pattern_name))]

        for col in metric_columns:
            metric_name = col.replace("mean_", "", 1)
            info = scores.get(metric_name) or {}
            mean = info.get("mean")
            if mean is not None:
                cell = "%.4f" % mean if isinstance(mean, (int, float)) else str(mean)
            else:
                cell = ""
            cells.append(cell)

        for col in config_columns:
            val = _get_config_value(merged, col) if merged else None
            if val is not None:
                cells.append(str(val))
            else:
                cells.append("")
        tr_class = ' class="rank-1"' if i == 0 else ""
        rows.append("<tr" + tr_class + ">" + "".join("<td>%s</td>" % html.escape(c) for c in cells) + "</tr>")

    table_body = "".join(rows)

    best_pattern_name = "—"
    if evaluations:
        best_pattern_name = (
            evaluations[0].get("name")
            or evaluations[0].get("pattern_name")
            or (evaluations[0].get("rag_pattern") or {}).get("name", "—")
        )
        best_pattern_name = str(best_pattern_name)

    html_content = _build_leaderboard_html(
        header_row=header_row,
        table_body=table_body,
        best_pattern_name=best_pattern_name,
        num_patterns=len(evaluations),
        eval_metric=optimization_metric or "faithfulness",
    )

    Path(html_artifact.path).parent.mkdir(parents=True, exist_ok=True)
    with open(html_artifact.path, "w", encoding="utf-8") as f:
        f.write(html_content)


if __name__ == "__main__":
    from kfp.compiler import Compiler

    Compiler().compile(
        leaderboard_evaluation,
        package_path=__file__.replace(".py", "_component.yaml"),
    )
