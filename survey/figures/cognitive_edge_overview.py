"""Draw the conceptual overview of reasoning-preserving cognitive edge computing.

This is a schematic (not a quantitative plot). It maps an edge task and its
budgets to a cross-layer controller, deployment tiers, and measured outcomes.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle


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


def line(ax, xs, ys, color, lw=0.85):
    ax.plot(xs, ys, color=color, lw=lw, solid_capstyle="round", zorder=4)


def pictogram(ax, kind, x, y, color, scale=1.0):
    """Draw a small original category mark; no vendor marks imply test coverage."""
    if kind == "chip":
        ax.add_patch(Rectangle((x - .10*scale, y - .075*scale), .20*scale, .15*scale,
                               facecolor="white", edgecolor=color, lw=.8, zorder=5))
        for d in (-.045, 0, .045):
            line(ax, [x + d*scale, x + d*scale], [y + .075*scale, y + .105*scale], color, .75)
            line(ax, [x + d*scale, x + d*scale], [y - .075*scale, y - .105*scale], color, .75)
            line(ax, [x - .10*scale, x - .13*scale], [y + d*scale, y + d*scale], color, .75)
            line(ax, [x + .10*scale, x + .13*scale], [y + d*scale, y + d*scale], color, .75)
    elif kind == "server":
        for dy in (-.052, .052):
            ax.add_patch(FancyBboxPatch((x - .13*scale, y + (dy - .031)*scale),
                                        .26*scale, .062*scale,
                                        boxstyle="round,pad=0.005,rounding_size=0.012",
                                        facecolor="white", edgecolor=color, lw=.75, zorder=5))
            ax.add_patch(Circle((x - .085*scale, y + dy*scale), .012*scale,
                                facecolor=color, edgecolor=color, lw=.4, zorder=6))
            line(ax, [x - .04*scale, x + .095*scale], [y + dy*scale]*2, color, .55)
    elif kind == "agent":
        pts = [(x - .09*scale, y - .05*scale), (x, y + .075*scale), (x + .09*scale, y - .05*scale)]
        line(ax, [pts[0][0], pts[1][0], pts[2][0]], [pts[0][1], pts[1][1], pts[2][1]], color, .85)
        for px, py in pts:
            ax.add_patch(Circle((px, py), .027*scale, facecolor="white", edgecolor=color, lw=.8, zorder=6))
    elif kind == "phone":
        ax.add_patch(FancyBboxPatch((x - .055*scale, y - .10*scale), .11*scale, .20*scale,
                                    boxstyle="round,pad=0.01,rounding_size=0.025",
                                    facecolor="white", edgecolor=color, lw=.75, zorder=5))
        line(ax, [x - .025*scale, x + .025*scale], [y - .07*scale]*2, color, .55)
    elif kind == "cloud":
        for cx, cy, radius in ((x-.045*scale, y, .05), (x, y+.035*scale, .065), (x+.055*scale, y, .05)):
            ax.add_patch(Circle((cx, cy), radius*scale, facecolor="white", edgecolor=color, lw=.7, zorder=5))
        ax.add_patch(Rectangle((x-.07*scale, y-.045*scale), .14*scale, .045*scale,
                               facecolor="white", edgecolor="none", zorder=5))
        line(ax, [x-.07*scale, x+.07*scale], [y-.045*scale]*2, color, .7)
    elif kind == "quality":
        ax.add_patch(Circle((x, y), .09*scale, facecolor="white", edgecolor=color, lw=.8, zorder=5))
        line(ax, [x-.045*scale, x-.01*scale, x+.05*scale],
             [y, y-.035*scale, y+.045*scale], color, 1.0)
    elif kind == "clock":
        ax.add_patch(Circle((x, y), .09*scale, facecolor="white", edgecolor=color, lw=.8, zorder=5))
        line(ax, [x, x], [y, y+.052*scale], color, .8)
        line(ax, [x, x+.04*scale], [y, y-.025*scale], color, .8)
    elif kind == "energy":
        pts = [(x+.015*scale,y+.10*scale),(x-.055*scale,y-.005*scale),
               (x-.005*scale,y-.005*scale),(x-.025*scale,y-.10*scale),
               (x+.06*scale,y+.015*scale),(x+.01*scale,y+.015*scale)]
        ax.add_patch(Polygon(pts, closed=True, facecolor="white", edgecolor=color, lw=.8, zorder=5))
    elif kind == "shield":
        pts = [(x-.075*scale,y+.065*scale),(x+.075*scale,y+.065*scale),
               (x+.06*scale,y-.015*scale),(x,y-.10*scale),(x-.06*scale,y-.015*scale)]
        ax.add_patch(Polygon(pts, closed=True, facecolor="white", edgecolor=color, lw=.8, zorder=5))
        line(ax, [x-.03*scale,x,x+.04*scale], [y-.005*scale,y-.04*scale,y+.03*scale], color, .8)


def box(ax, x, y, w, h, title, body, edge, face, title_color=INK, icon=None):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.035,rounding_size=0.08",
        linewidth=0.9, edgecolor=edge, facecolor=face,
    )
    ax.add_patch(patch)
    title_x = x + w / 2
    if icon:
        pictogram(ax, icon, x + .44, y + h * .76, edge, scale=.72)
        title_x += .12
    ax.text(title_x, y + h * 0.76, title, ha="center", va="center",
            color=title_color, weight="bold", fontsize=7.0)
    if body:
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
box(ax, xs[0], card_y, card_w, card_h, "MODEL", "precision · size · experts", BLUE, PALE_BLUE, icon="chip")
box(ax, xs[1], card_y, card_w, card_h, "RUNTIME", "cache · schedule · offload", TEAL, PALE_TEAL, icon="server")
box(ax, xs[2], card_y, card_w, card_h, "AGENT", "context · tools · depth", GOLD, PALE_GOLD, icon="agent")

box(ax, 1.02, 1.94, 7.96, 1.10,
    "TIERED EXECUTION · LINK-AWARE PLACEMENT", "", TEAL, PALE_TEAL)
for xx, kind, label, color in ((2.35,"phone","DEVICE",BLUE),
                               (5.0,"server","EDGE",TEAL),
                               (7.65,"cloud","CLOUD",GOLD)):
    pictogram(ax, kind, xx - .22, 2.25, color, scale=.82)
    ax.text(xx + .05, 2.25, label, ha="left", va="center", color=INK,
            fontsize=5.8, weight="bold")
arrow(ax, (3.38,2.25), (4.27,2.25), color=LINE, lw=.9, ms=7)
arrow(ax, (6.03,2.25), (6.92,2.25), color=LINE, lw=.9, ms=7)
for x in (2.15, 5.0, 7.85):
    arrow(ax, (x, card_y - 0.04), (x, 3.06), color=TEAL, lw=0.9, ms=7)

box(ax, 1.02, 0.39, 7.96, 1.10,
    "OBSERVED OUTCOMES",
    "",
    GOLD, PALE_GOLD)
outcomes = ((2.0,"quality","quality",TEAL),(4.0,"clock","p95 latency",BLUE),
            (6.0,"energy","energy / token",GOLD),(8.0,"shield","safety",TEAL))
for xx, kind, label, color in outcomes:
    pictogram(ax, kind, xx, .79, color, scale=.78)
    ax.text(xx, .55, label, ha="center", va="center", color=MUTED, fontsize=5.8)
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
