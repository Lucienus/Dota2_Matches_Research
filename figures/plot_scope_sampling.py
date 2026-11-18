"""Render manuscript Figure 4 from its frozen table of means."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_scope_sampling(rows):
    positions = range(7)
    targets = [
        ("all", "Full target (7,541 matches)", "#28628A", "o", -0.07),
        ("listed", "Listed target (1,502 matches)", "#A75820", "s", 0.07),
    ]
    values = {(r["target"], int(r["position"])): r for r in rows}
    expected = {(target[0], pos) for target in targets for pos in positions}
    if len(rows) != len(expected) or values.keys() != expected:
        raise ValueError("Expected two targets for the original references and five new draws.")
    if any(int(r["seeds"]) != (10 if pos == 0 else 3) for (_, pos), r in values.items()):
        raise ValueError("Seed counts differ from the original and repeated-draw design.")

    fig, ax = plt.subplots(figsize=(7.2, 3.35))
    fig.subplots_adjust(left=0.13, right=0.98, top=0.84, bottom=0.19)
    ax.axvspan(-0.5, 1.5, color="#EDF1F4", zorder=0)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#DCDFE2", linewidth=0.55)
    ax.axhline(0, color="#384650", linewidth=0.95)
    for target, label, color, marker, offset in targets:
        ax.scatter([pos + offset for pos in positions],
                   [float(values[target, pos]["mean_difference"]) for pos in positions],
                   color=color, marker=marker, s=34, label=label, zorder=3)
    ax.set_xlim(-0.5, 6.5)
    ax.set_ylim(-0.0041, 0.0051)
    ax.set_yticks([-0.004, -0.002, 0, 0.002, 0.004])
    ax.set_xticks(positions, ["Original\n10 seeds", "Original\n3 seeds",
                              *[f"Draw {i}\n3 seeds" for i in range(1, 6)]])
    ax.set_ylabel("Listed − matched log loss (nats)")
    ax.spines[["right", "top"]].set_visible(False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.04), ncol=2, frameon=False)
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("output_path", type=Path)
    args = parser.parse_args()
    with args.csv_path.open(encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 8,
                         "pdf.fonttype": 42, "svg.fonttype": "none", "axes.linewidth": 0.7}):
        figure = plot_scope_sampling(rows)
        metadata = {"CreationDate": None} if args.output_path.suffix.lower() == ".pdf" else {}
        figure.savefig(args.output_path, metadata=metadata)
        plt.close(figure)
