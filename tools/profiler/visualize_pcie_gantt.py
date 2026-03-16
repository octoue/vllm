# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Visualize PCIe transfer events as an interactive Gantt chart.

Reads pcie_events.json from PCIeTracer and generates an HTML file with
plotly showing PCIe bandwidth usage over time.

Requires: pip install plotly
"""

import argparse
import json
from pathlib import Path

import plotly.graph_objects as go

# Color map: Offload-Evict (D2H), Offload-Restore (H2D), PP-Transfer, Prefetch
COLOR_MAP = {
    "Evict": "#E74C3C",
    "Restore": "#E67E22",
    "PP_Transfer": "#3498DB",
    "Prefetch": "#2ECC71",
}


def load_events(json_path: str) -> list[dict]:
    """Load PCIe events from JSON file."""
    with open(json_path) as f:
        return json.load(f)


def build_gantt(events: list[dict], output_path: str) -> None:
    """Build interactive Gantt chart and save to HTML."""
    if not events:
        print("No events to visualize.")
        return

    # Normalize time to start from 0
    min_start_us = min(e["start_us"] for e in events)
    for e in events:
        e["start_ms_norm"] = (e["start_us"] - min_start_us) / 1000.0

    fig = go.Figure()

    for event in events:
        lane = f"GPU{event['gpu_id']}_{event['direction']}"
        color = COLOR_MAP.get(event["op_type"], "#95A5A6")

        fig.add_trace(
            go.Bar(
                name=event["op_type"],
                y=[lane],
                x=[event["duration_ms"]],
                base=[event["start_ms_norm"]],
                orientation="h",
                marker_color=color,
                hovertemplate=(
                    f"<b>Type</b>: {event['op_type']}<br>"
                    f"<b>Size</b>: {event['size_mb']:.2f} MB<br>"
                    f"<b>Duration</b>: {event['duration_ms']:.3f} ms<br>"
                    f"<b>Bandwidth</b>: {event['bandwidth_gbps']:.2f} GB/s"
                    "<extra></extra>"
                ),
                showlegend=False,
            )
        )

    # Sort lanes: GPU0_H2D, GPU0_D2H, GPU1_H2D, GPU1_D2H, ...
    all_lanes = sorted(
        set(f"GPU{e['gpu_id']}_{e['direction']}" for e in events),
        key=lambda s: (int(s.split("GPU")[1].split("_")[0]), s.split("_")[1]),
    )
    # Total timeline span - bars can be invisible if events are short vs total range
    max_time_ms = max(e["start_ms_norm"] + e["duration_ms"] for e in events)
    # Default view: first 5 seconds or 10% of total, whichever is larger (min 500ms)
    default_x_range = min(5000.0, max(500.0, max_time_ms * 0.1))

    fig.update_layout(
        yaxis=dict(
            categoryorder="array",
            categoryarray=all_lanes,
            title="PCIe Lane",
        ),
        xaxis=dict(
            title="Time (ms)",
            type="linear",
            range=[0, default_x_range],
            rangeslider=dict(
                visible=True,
                range=[0, max_time_ms],
                thickness=0.05,
            ),
        ),
        title="PCIe Bandwidth Usage (use rangeslider below to zoom/pan)",
        barmode="overlay",
        height=400 + len(all_lanes) * 30,
        margin=dict(l=120),
        hovermode="closest",
    )

    # Add legend for op types (invisible traces)
    for op_type, color in COLOR_MAP.items():
        if any(e["op_type"] == op_type for e in events):
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    marker=dict(size=10, color=color, symbol="square"),
                    name=op_type,
                    showlegend=True,
                )
            )

    fig.write_html(output_path)
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
