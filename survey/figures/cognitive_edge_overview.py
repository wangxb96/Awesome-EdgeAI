"""Draw the conceptual overview of reasoning-preserving cognitive edge computing.

This is a schematic (not a quantitative plot). It maps an edge task and its
budgets to a cross-layer controller, deployment tiers, and measured outcomes.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUT = Path(__file__).resolve().parent
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 7,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})

INK = "#263746"
MUTED = "#536878"
BLUE = "#4D7EA8"
PALE_BLUE = "#EAF1F6"
TEAL = "#4E8C87"
PALE_TEAL = "#EAF3F1"
GOLD = "#B47B35"
PALE_GOLD = "#F7F0E5"
LINE = "#9AAAB5"


def box(ax, x, y, w, h, title, body, edge, face, title_color=INK):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.035,rounding_size=0.08",
        linewidth=0.9, edgecolor=edge, facecolor=face,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h * 0.76, title, ha="center", va="center",
            color=title_color, weight="bold", fontsize=7.0)
    ax.text(x + w / 2, y + h * 0.27, body, ha="center", va="center",
            color=MUTED, fontsize=5.8, linespacing=1.3)
    return patch


def arrow(ax, start, end, color=LINE, rad=0, lw=1.0, ms=8):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=ms,
        linewidth=lw, color=color,
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=1.5, shrinkB=1.5,
    ))


fig, ax = plt.subplots(figsize=(3.48, 2.23))
fig.subplots_adjust(left=0.015, right=0.985, bottom=0.045, top=0.97)
ax.set_xlim(0, 10)
ax.set_ylim(0, 6.35)
ax.axis("off")

# Task and system budgets enter a closed-loop policy.
box(ax, 2.15, 5.05, 5.7, 0.93,
    "EDGE TASK + BUDGETS",
    "reasoning goal  ·  latency · energy · memory · link",
    BLUE, PALE_BLUE)

ax.text(5, 4.75, "CROSS-LAYER ADAPTATION", ha="center", va="center",
        color=INK, fontsize=7.0, weight="bold")
card_y, card_h, card_w = 3.40, 1.10, 2.75
xs = [0.38, 3.625, 6.87]
box(ax, xs[0], card_y, card_w, card_h, "MODEL", "precision · size · experts", BLUE, PALE_BLUE)
box(ax, xs[1], card_y, card_w, card_h, "RUNTIME", "cache · schedule · offload", TEAL, PALE_TEAL)
box(ax, xs[2], card_y, card_w, card_h, "AGENT", "context · tools · depth", GOLD, PALE_GOLD)

box(ax, 1.02, 1.94, 7.96, 1.10,
    "EXECUTION ACROSS TIERS",
    "DEVICE     ↔     EDGE     ↔     CLOUD     ·     link-aware placement",
    TEAL, PALE_TEAL)
for x in (2.15, 5.0, 7.85):
    arrow(ax, (x, card_y - 0.04), (x, 3.06), color=TEAL, lw=0.9, ms=7)

box(ax, 1.02, 0.39, 7.96, 1.10,
    "OBSERVED OUTCOMES",
    "reasoning quality · tail latency · energy / token · safety",
    GOLD, PALE_GOLD)
arrow(ax, (5, 1.91), (5, 1.52), color=GOLD, lw=1.1)

# Feedback returns measured outcomes to the policy, without crossing labels.
arrow(ax, (8.99, 0.92), (9.82, 0.92), color=BLUE, lw=1.0, ms=8)
arrow(ax, (9.82, 0.92), (9.82, 5.49), color=BLUE, rad=0.0, lw=1.0, ms=8)
arrow(ax, (9.82, 5.49), (7.93, 5.49), color=BLUE, lw=1.0, ms=8)

fig.savefig(OUT / "cognitive_edge_overview.svg", bbox_inches="tight", pad_inches=0.02)
fig.savefig(OUT / "cognitive_edge_overview.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(OUT / "cognitive_edge_overview.tiff", dpi=600, bbox_inches="tight",
            pad_inches=0.02, pil_kwargs={"compression": "tiff_lzw"})
svg_path = OUT / "cognitive_edge_overview.svg"
svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n")
plt.close(fig)
