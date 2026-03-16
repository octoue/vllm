# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Visualize PCIe transfer events as an interactive Gantt chart.

Reads pcie_events.json from PCIeTracer and generates an HTML file with
plotly showing PCIe bandwidth usage over time. Includes:
- Event cluster detection and zoom buttons
- Density overview timeline
- Statistical summary panel
- Merged traces for better performance

Requires: pip install plotly
"""

import argparse
import json
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Color map: Offload-Evict (D2H), Offload-Restore (H2D), PP-Transfer, Prefetch
COLOR_MAP = {
    "Evict": "#E74C3C",
    "Restore": "#E67E22",
    "PP_Transfer": "#3498DB",
    "Prefetch": "#2ECC71",
}

# Gap threshold (ms) to split events into clusters
CLUSTER_GAP_MS = 2000.0

# Overview bin size (ms)
OVERVIEW_BIN_MS = 1000.0


def load_events(json_path: str) -> list[dict]:
    """Load PCIe events from JSON file."""
    with open(json_path) as f:
        return json.load(f)


def compute_stats(events: list[dict]) -> dict:
    """Compute per-op_type statistics."""
    stats = {}
    for e in events:
        op = e["op_type"]
        if op not in stats:
            stats[op] = {
                "count": 0,
                "total_bytes": 0,
                "total_duration_ms": 0.0,
                "bandwidths": [],
            }
        stats[op]["count"] += 1
        stats[op]["total_bytes"] += e["size_bytes"]
        stats[op]["total_duration_ms"] += e["duration_ms"]
        stats[op]["bandwidths"].append(e["bandwidth_gbps"])

    for op, s in stats.items():
        s["total_mb"] = s["total_bytes"] / (1024 * 1024)
        s["avg_duration_ms"] = (
            s["total_duration_ms"] / s["count"] if s["count"] else 0
        )
        s["avg_bandwidth_gbps"] = (
            sum(s["bandwidths"]) / len(s["bandwidths"])
            if s["bandwidths"]
            else 0
        )
    return stats


def format_stats_html(stats: dict, total_span_ms: float) -> str:
    """Format statistics as HTML string."""
    lines = []
    lines.append("<table border='1' cellpadding='6' style='border-collapse: collapse; font-size: 13px;'>")
    lines.append(
        "<tr><th>Op Type</th><th>Count</th><th>Total (MB)</th>"
        "<th>Avg Duration (ms)</th><th>Avg BW (GB/s)</th></tr>"
    )
    for op, s in stats.items():
        lines.append(
            f"<tr><td>{op}</td><td>{s['count']}</td><td>{s['total_mb']:.1f}</td>"
            f"<td>{s['avg_duration_ms']:.2f}</td><td>{s['avg_bandwidth_gbps']:.2f}</td></tr>"
        )
    total_active_ms = sum(s["total_duration_ms"] for s in stats.values())
    utilization_pct = (total_active_ms / total_span_ms * 100) if total_span_ms > 0 else 0
    lines.append("</table>")
    lines.append(
        f"<p><b>Total span:</b> {total_span_ms/1000:.2f} s | "
        f"<b>Active transfer time:</b> {total_active_ms:.1f} ms | "
        f"<b>PCIe utilization:</b> {utilization_pct:.3f}%</p>"
    )
    return "\n".join(lines)


def detect_clusters(events: list[dict]) -> list[tuple[float, float]]:
    """Detect event clusters. Returns list of (start_ms, end_ms) per cluster."""
    if not events:
        return []
    sorted_events = sorted(events, key=lambda e: e["start_ms_norm"])
    clusters = []
    cluster_start = sorted_events[0]["start_ms_norm"]
    cluster_end = cluster_start + sorted_events[0]["duration_ms"]

    for e in sorted_events[1:]:
        start = e["start_ms_norm"]
        end = start + e["duration_ms"]
        if start - cluster_end <= CLUSTER_GAP_MS:
            cluster_end = max(cluster_end, end)
        else:
            clusters.append((cluster_start, cluster_end))
            cluster_start = start
            cluster_end = end
    clusters.append((cluster_start, cluster_end))
    return clusters


def build_overview_bins(events: list[dict], max_time_ms: float) -> tuple[list[float], list[float]]:
    """Build overview: (bin_centers, data_per_bin in MB)."""
    num_bins = max(1, int(max_time_ms / OVERVIEW_BIN_MS) + 1)
    bins = [0.0] * num_bins
    for e in events:
        start_bin = int(e["start_ms_norm"] / OVERVIEW_BIN_MS)
        end_bin = int((e["start_ms_norm"] + e["duration_ms"]) / OVERVIEW_BIN_MS)
        size_mb = e["size_mb"]
        for b in range(start_bin, min(end_bin + 1, num_bins)):
            bins[b] += size_mb
    centers = [(i + 0.5) * OVERVIEW_BIN_MS for i in range(num_bins)]
    return centers, bins


def build_gantt(events: list[dict], output_path: str) -> None:
    """Build interactive Gantt chart with overview, clusters, and stats."""
    if not events:
        print("No events to visualize.")
        return

    # Normalize time to start from 0
    min_start_us = min(e["start_us"] for e in events)
    for e in events:
        e["start_ms_norm"] = (e["start_us"] - min_start_us) / 1000.0

    max_time_ms = max(e["start_ms_norm"] + e["duration_ms"] for e in events)
    total_span_ms = max_time_ms

    # Statistics
    stats = compute_stats(events)
    stats_html = format_stats_html(stats, total_span_ms)

    # Clusters
    clusters = detect_clusters(events)

    # Overview bins
    centers, bin_data = build_overview_bins(events, max_time_ms)

    # Lanes
    all_lanes = sorted(
        set(f"GPU{e['gpu_id']}_{e['direction']}" for e in events),
        key=lambda s: (int(s.split("GPU")[1].split("_")[0]), s.split("_")[1]),
    )

    # Create subplots: row 1 = overview, row 2 = gantt
    fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.25, 0.75],
        vertical_spacing=0.08,
        subplot_titles=(
            "PCIe Data Volume per Second (Overview - click to zoom)",
            "PCIe Transfer Gantt (Evict=red, Restore=orange, Prefetch=green, PP=blue)",
        ),
        specs=[[{"type": "bar"}], [{"type": "bar"}]],
    )

    # Overview trace
    fig.add_trace(
        go.Bar(
            x=centers,
            y=bin_data,
            marker_color="#7F8C8D",
            opacity=0.7,
            name="Data (MB/s)",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.update_xaxes(title_text="Time (ms)", row=1, col=1)
    fig.update_yaxes(title_text="MB", row=1, col=1)

    # Gantt: merge events by op_type into one trace per type
    for op_type, color in COLOR_MAP.items():
        group = [e for e in events if e["op_type"] == op_type]
        if not group:
            continue
        fig.add_trace(
            go.Bar(
                x=[e["duration_ms"] for e in group],
                y=[f"GPU{e['gpu_id']}_{e['direction']}" for e in group],
                base=[e["start_ms_norm"] for e in group],
                orientation="h",
                marker_color=color,
                name=op_type,
                legendgroup=op_type,
                hovertemplate=(
                    "<b>Type</b>: " + op_type + "<br>"
                    "<b>Size</b>: %{customdata[0]:.2f} MB<br>"
                    "<b>Duration</b>: %{x:.3f} ms<br>"
                    "<b>Bandwidth</b>: %{customdata[1]:.2f} GB/s"
                    "<extra></extra>"
                ),
                customdata=[[e["size_mb"], e["bandwidth_gbps"]] for e in group],
            ),
            row=2,
            col=1,
        )

    fig.update_xaxes(
        title_text="Time (ms)",
        row=2,
        col=1,
        range=[0, min(10000.0, max(1000.0, max_time_ms * 0.08))],
        rangeslider=dict(
            visible=True,
            range=[0, max_time_ms],
            thickness=0.06,
        ),
    )
    fig.update_yaxes(
        categoryorder="array",
        categoryarray=all_lanes,
        title_text="PCIe Lane",
        row=2,
        col=1,
    )

    # Build updatemenus for cluster zoom buttons
    buttons = []
    # Button: Show all
    buttons.append(
        dict(
            label="Show All",
            method="relayout",
            args=[
                {
                    "xaxis2.range": [0, max_time_ms],
                    "xaxis2.rangeslider.range": [0, max_time_ms],
                }
            ],
        )
    )
    for i, (c_start, c_end) in enumerate(clusters):
        pad = max(500.0, (c_end - c_start) * 0.2)
        view_start = max(0, c_start - pad)
        view_end = min(max_time_ms, c_end + pad)
        buttons.append(
            dict(
                label=f"Cluster {i + 1} ({c_start/1000:.1f}s - {c_end/1000:.1f}s)",
                method="relayout",
                args=[
                    {
                        "xaxis2.range": [view_start, view_end],
                        "xaxis2.rangeslider.range": [0, max_time_ms],
                    }
                ],
            )
        )

    fig.update_layout(
        barmode="overlay",
        height=550 + len(all_lanes) * 25,
        margin=dict(l=120, t=80, b=80),
        hovermode="closest",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        updatemenus=[
            dict(
                type="buttons",
                direction="down",
                showactive=True,
                x=0.01,
                y=1.08,
                xanchor="left",
                yanchor="top",
                buttons=buttons,
            )
        ],
    )

    # Write HTML with stats panel
    fig_html = fig.to_html(full_html=False, include_plotlyjs="cdn")
    full_html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>PCIe Gantt Chart</title>
  <style>
    #stats {{ font-family: sans-serif; padding: 12px; margin: 10px; background: #f8f9fa; border-radius: 6px; }}
    #stats h3 {{ margin-top: 0; }}
  </style>
</head>
<body>
<div id="stats">
  <h3>PCIe Transfer Summary</h3>
  {stats_html}
  <p><b>Clusters detected:</b> {len(clusters)} (use buttons above chart to zoom)</p>
</div>
{fig_html}
</body>
</html>"""

    with open(output_path, "w") as f:
        f.write(full_html)
    print(f"Saved Gantt chart to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize PCIe transfer events as Gantt chart"
    )
    parser.add_argument(
        "--input",
        "-i",
        default="pcie_events.json",
        help="Input JSON file from PCIeTracer",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="pcie_gantt.html",
        help="Output HTML file",
    )
    args = parser.parse_args()

    events = load_events(args.input)
    build_gantt(events, args.output)


if __name__ == "__main__":
    main()
