#!/usr/bin/env python3
"""Generate the deterministic evidence-architecture figure for CRN-2026-003.

The diagram is deliberately outcome-neutral.  It separates the archived,
node-local policy audit from the prospective shared-channel packet experiment
and states the independent-unit counts without implying that comparator runs
were replicated over training seeds.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402


STUDY_ROOT: Final = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT: Final = STUDY_ROOT.parent.parent
DEFAULT_OUTPUT: Final = WORKSPACE_ROOT / "main/manuscript/figures/system_architecture.pdf"
WIDTH_IN: Final = 7.16
HEIGHT_IN: Final = 3.55
FIXED_TIME: Final = datetime(2026, 10, 20, 0, 0, 0, tzinfo=UTC)

BLUE: Final = "#2767A5"
BLUE_FILL: Final = "#E8F0FA"
GREEN: Final = "#26734D"
GREEN_FILL: Final = "#E7F4EC"
PURPLE: Final = "#6B3FA0"
PURPLE_FILL: Final = "#F0E8F8"
ORANGE: Final = "#A65316"
ORANGE_FILL: Final = "#FCEBDB"
TEXT: Final = "#17212B"


def _configure() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "path.simplify": False,
            "savefig.transparent": False,
            "savefig.facecolor": "white",
        }
    )


def _box(ax, x: float, y: float, width: float, height: float, *, title: str,
         body: str, edge: str, fill: str, dashed: bool = False,
         double: bool = False) -> tuple[float, float, float, float]:
    patch = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.012,rounding_size=0.035",
        linewidth=1.0,
        linestyle=(0, (3, 1.8)) if dashed else "solid",
        edgecolor=edge,
        facecolor=fill,
        zorder=2,
    )
    ax.add_patch(patch)
    if double:
        inset = 0.026
        ax.add_patch(
            FancyBboxPatch(
                (x + inset, y + inset), width - 2 * inset, height - 2 * inset,
                boxstyle="round,pad=0.005,rounding_size=0.028",
                linewidth=0.55, edgecolor=edge, facecolor="none", zorder=3,
            )
        )
    ax.text(x + width / 2, y + height - 0.12, title, ha="center", va="center",
            fontsize=6.6, fontweight="bold", color=edge, zorder=4)
    ax.text(x + width / 2, y + height * 0.42, body, ha="center", va="center",
            fontsize=6.65, color=TEXT, linespacing=1.04, zorder=4)
    return x, y, width, height


def _arrow(ax, source: tuple[float, float, float, float],
           target: tuple[float, float, float, float], color: str,
           dashed: bool = False) -> None:
    sx, sy, sw, sh = source
    tx, ty, _, th = target
    ax.add_patch(
        FancyArrowPatch(
            (sx + sw, sy + sh / 2), (tx, ty + th / 2),
            arrowstyle="-|>", mutation_scale=7.0, linewidth=0.95,
            linestyle=(0, (3, 1.8)) if dashed else "solid",
            color=color, shrinkA=1.2, shrinkB=1.2, zorder=5,
        )
    )


def _lane(ax, y: float, *, edge: str, fill: str, label: str) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (0.03, y), 7.10, 1.48,
            boxstyle="round,pad=0.012,rounding_size=0.05",
            linewidth=1.05, edgecolor=edge, facecolor=fill, zorder=0,
        )
    )
    ax.text(0.13, y + 1.31, label, ha="left", va="center", fontsize=8.0,
            fontweight="bold", color=edge, zorder=4)


def render() -> bytes:
    _configure()
    fig, ax = plt.subplots(figsize=(WIDTH_IN, HEIGHT_IN))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_xlim(0, WIDTH_IN)
    ax.set_ylim(0, HEIGHT_IN)
    ax.axis("off")

    _lane(
        ax, 1.91, edge=BLUE, fill="#F7FAFE",
        label="A  Archived diagnostic: node-local emulator; no secondary-user collision rule",
    )
    xs = (0.13, 1.23, 2.37, 3.55, 4.77, 5.82)
    widths = (1.00, 1.04, 1.08, 1.12, 0.95, 1.18)
    top = [
        _box(ax, xs[0], 2.18, widths[0], 0.88, title="FROZEN INPUTS",
             body="Independent\nnode-local traces", edge=BLUE, fill=BLUE_FILL),
        _box(ax, xs[1], 2.18, widths[1], 0.88, title="STATE / MASK",
             body="128 features\n+ legal channels", edge=BLUE, fill=BLUE_FILL),
        _box(ax, xs[2], 2.18, widths[2], 0.88, title="EXECUTABLE",
             body="Checkpoint scores\n$\\rightarrow$ masked greedy", edge=BLUE, fill=BLUE_FILL),
        _box(ax, xs[3], 2.18, widths[3], 0.88, title="ACTION RECORDS",
             body="Discovery + disjoint\nvalidation cohorts", edge=BLUE, fill=BLUE_FILL),
        _box(ax, xs[4], 2.18, widths[4], 0.88, title="EXACT AUDIT",
             body="63 masks +\nprecedence graph", edge=PURPLE, fill=PURPLE_FILL, dashed=True),
        _box(ax, xs[5], 2.18, widths[5], 0.88, title="BOUNDED CLAIM",
             body="Finite-support\ncertificate +\noff-support search", edge=PURPLE,
             fill=PURPLE_FILL, dashed=True, double=True),
    ]
    for index in range(3):
        _arrow(ax, top[index], top[index + 1], BLUE)
    _arrow(ax, top[3], top[4], PURPLE, dashed=True)
    _arrow(ax, top[4], top[5], PURPLE, dashed=True)
    ax.text(0.14, 2.00, "Independent unit: checkpoint; certificate applies only to the declared finite records.",
            ha="left", va="center", fontsize=6.55, color=TEXT, fontstyle="italic")

    _lane(
        ax, 0.18, edge=GREEN, fill="#F7FCF9",
        label="B  Prospective experiment: common availability, persistent queues, and destructive collisions",
    )
    bottom = [
        _box(ax, xs[0], 0.44, widths[0], 0.88, title="FROZEN DESIGN",
             body="Seeds, factors,\ncode + dependencies", edge=GREEN, fill=GREEN_FILL),
        _box(ax, xs[1], 0.44, widths[1], 0.88, title="COMMON STATE",
             body="Availability +\npacket queues", edge=GREEN, fill=GREEN_FILL),
        _box(ax, xs[2], 0.44, widths[2], 0.88, title="JOINT ACTIONS",
             body="3 DQN variants\n+ comparators", edge=GREEN, fill=GREEN_FILL),
        _box(ax, xs[3], 0.44, widths[3], 0.88, title="OUTCOME KERNEL",
             body="$\\geq2$: collision\nsingleton: link trial", edge=ORANGE, fill=ORANGE_FILL),
        _box(ax, xs[4], 0.44, widths[4], 0.88, title="PACKET STATE",
             body="Delivery or\nrequeue", edge=GREEN, fill=GREEN_FILL),
        _box(ax, xs[5], 0.44, widths[5], 0.88, title="GOLD EVIDENCE",
             body="DQN: 20 seeds $\\times$\n8 cells $\\times$ 24 blocks\n+ claims and hashes",
             edge=PURPLE, fill=PURPLE_FILL, dashed=True, double=True),
    ]
    for index in range(4):
        _arrow(ax, bottom[index], bottom[index + 1], GREEN)
    _arrow(ax, bottom[4], bottom[5], PURPLE, dashed=True)
    ax.text(0.14, 0.27, "Comparators: 8 cells $\\times$ 24 blocks; training seed is the DQN independent unit.",
            ha="left", va="center", fontsize=6.55, color=TEXT, fontstyle="italic")

    stream = io.BytesIO()
    fig.savefig(
        stream, format="pdf", dpi=300,
        metadata={
            "Title": "ChainRadioNet-AI evidence architecture",
            "Author": "ChainRadioNet-AI authors",
            "Subject": "Outcome-neutral separation of archived and prospective studies",
            "Creator": "generate_architecture_figure.py",
            "CreationDate": FIXED_TIME,
            "ModDate": FIXED_TIME,
        },
    )
    plt.close(fig)
    return stream.getvalue()


def _write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            os.fchmod(handle.fileno(), 0o644)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    payload = render()
    output = args.output.resolve()
    if args.check:
        if not output.is_file() or output.read_bytes() != payload:
            raise SystemExit(f"stale or non-deterministic generated artifact: {output}")
    else:
        _write(output, payload)
    print(f"{output}: {hashlib.sha256(payload).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
