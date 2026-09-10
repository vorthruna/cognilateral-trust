"""Leaderboard generator — self-contained HTML with inline CSS."""

from __future__ import annotations

from html import escape

from cognilateral_trust.bench.scoring import BenchScore

__all__ = ["generate_leaderboard"]


def generate_leaderboard(results: list[BenchScore] | tuple[BenchScore, ...]) -> str:
    """Generate a self-contained HTML leaderboard.

    Renders a table with model rankings, overall score, and per-domain scores.
    Pre-sorted by overall score (descending).
    No external resources — all CSS inline, no network requests.

    Args:
        results: Sequence of BenchScore objects

    Returns:
        Self-contained HTML string
    """
    # Sort by overall_score descending
    sorted_results = sorted(results, key=lambda r: r.overall_score, reverse=True)

    # Collect all unique domains
    all_domains: set[str] = set()
    for result in sorted_results:
        for ds in result.domain_scores:
            all_domains.add(ds.domain)
    sorted_domains = sorted(all_domains)

    # Build table rows
    table_rows = ""
    for rank, result in enumerate(sorted_results, 1):
        # Render per-domain calibration as a score (1.0 - ECE) so every number in
        # the table is higher-is-better, consistent with the overall score and footer.
        domain_scores_dict = {ds.domain: (1.0 - ds.calibration_error) for ds in result.domain_scores}

        row_html = f"""    <tr>
      <td class="rank">{rank}</td>
      <td class="model">{escape(result.model)}</td>
      <td class="overall-score">{result.overall_score:.4f}</td>
"""
        for domain in sorted_domains:
            domain_score = domain_scores_dict.get(domain, 0.0)
            row_html += f'      <td class="domain-score">{domain_score:.4f}</td>\n'

        row_html += "    </tr>\n"
        table_rows += row_html

    # Build table header
    header_cells = (
        '<th class="rank">Rank</th>\n      <th class="model">Model</th>\n      <th class="overall-score">Overall</th>\n'
    )
    for domain in sorted_domains:
        header_cells += f'      <th class="domain-header">{escape(domain)}</th>\n'

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>TrustBench Leaderboard</title>
  <style>
    * {{
      margin: 0;
      padding: 0;
      box-sizing: border-box;
    }}

    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      min-height: 100vh;
      padding: 2rem;
    }}

    .container {{
      max-width: 1200px;
      margin: 0 auto;
      background: white;
      border-radius: 8px;
      box-shadow: 0 10px 40px rgba(0, 0, 0, 0.2);
      overflow: hidden;
    }}

    .header {{
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      color: white;
      padding: 2rem;
      text-align: center;
    }}

    .header h1 {{
      font-size: 2rem;
      margin-bottom: 0.5rem;
    }}

    .header p {{
      font-size: 1rem;
      opacity: 0.9;
    }}

    .table-wrapper {{
      overflow-x: auto;
      padding: 2rem;
    }}

    table {{
      width: 100%;
      border-collapse: collapse;
    }}

    th {{
      background: #f7f8fc;
      padding: 1rem;
      text-align: left;
      font-weight: 600;
      color: #333;
      border-bottom: 2px solid #e0e0e0;
      font-size: 0.9rem;
    }}

    th.rank {{
      width: 60px;
      text-align: center;
    }}

    th.model {{
      width: 200px;
    }}

    th.overall-score {{
      width: 100px;
      text-align: center;
    }}

    th.domain-header {{
      width: 100px;
      text-align: center;
      font-size: 0.85rem;
    }}

    td {{
      padding: 1rem;
      border-bottom: 1px solid #e0e0e0;
      color: #333;
    }}

    tr:hover {{
      background: #f9f9f9;
    }}

    td.rank {{
      text-align: center;
      font-weight: 600;
      color: #667eea;
      width: 60px;
    }}

    td.model {{
      font-weight: 500;
      width: 200px;
    }}

    td.overall-score {{
      text-align: center;
      font-weight: 600;
      width: 100px;
      color: #333;
      background: #fffbf0;
    }}

    td.domain-score {{
      text-align: center;
      width: 100px;
      font-size: 0.95rem;
    }}

    /* Score color coding */
    td.overall-score {{
      color: #333;
    }}

    td.domain-score {{
      color: #666;
    }}

    .footer {{
      background: #f7f8fc;
      padding: 1rem 2rem;
      text-align: center;
      color: #666;
      font-size: 0.85rem;
      border-top: 1px solid #e0e0e0;
    }}

    @media (max-width: 768px) {{
      .header h1 {{
        font-size: 1.5rem;
      }}

      .table-wrapper {{
        padding: 1rem;
      }}

      th, td {{
        padding: 0.75rem 0.5rem;
        font-size: 0.9rem;
      }}

      th.model, td.model {{
        width: auto;
      }}
    }}
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h1>TrustBench Leaderboard</h1>
      <p>Model Epistemic Honesty Rankings — Calibration Score (1 − ECE)</p>
    </div>

    <div class="table-wrapper">
      <table>
        <thead>
          <tr>
            {header_cells}
          </tr>
        </thead>
        <tbody>
{table_rows}        </tbody>
      </table>
    </div>

    <div class="footer">
      <p>Higher scores indicate better calibration. Score = 1.0 - ECE (Expected Calibration Error).</p>
    </div>
  </div>
</body>
</html>
"""

    return html
