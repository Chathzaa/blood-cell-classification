import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
})

fig, ax = plt.subplots(figsize=(7.2, 4.1))
ax.set_xlim(0, 10.6)
ax.set_ylim(0, 5.9)
ax.axis("off")
ax.set_aspect("equal")

def box(x, y, w, h, text, fs=8.2, lw=1.0):
    r = Rectangle((x, y), w, h, linewidth=lw, edgecolor="black", facecolor="white", zorder=2)
    ax.add_patch(r)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs, zorder=3)

def arrow(x1, y1, x2, y2):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=9,
                         linewidth=0.9, color="black", zorder=1,
                         shrinkA=0, shrinkB=0)
    ax.add_patch(a)

def elbow(points):
    """Draw a polyline through points, arrowhead only on the last segment."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    ax.plot(xs[:-1], ys[:-1], color="black", linewidth=0.9, zorder=1)
    arrow(points[-2][0], points[-2][1], points[-1][0], points[-1][1])

# --- Row 1: detection / classification cascade ---
box(0.2, 4.3, 1.5, 1.0, "Raw smear\nimage")
box(2.05, 4.3, 2.05, 1.0, "Stage 1\nYOLOv8 detector\n(RBC/WBC/Platelet)")
box(4.45, 4.3, 1.55, 1.0, "WBC crops\n(padded)")
box(6.55, 4.85, 2.0, 0.72, "Stage 2\nWBC subtype (4-class)")
box(6.55, 3.8, 2.0, 0.72, "Stage 3\nALL severity (4-class)")

arrow(1.7, 4.8, 2.05, 4.8)
arrow(4.0, 4.8, 4.45, 4.8)
arrow(6.0, 4.95, 6.55, 5.15)
arrow(6.0, 4.65, 6.55, 4.25)

# --- Row 2: classical feature / rule pipeline ---
box(0.2, 2.6, 1.5, 1.0, "Preprocessing\nLAB + CLAHE +\ndenoise")
box(2.05, 2.6, 2.05, 1.0, "Otsu mask +\nmarker-controlled\nwatershed")
box(4.45, 2.6, 1.55, 1.0, "22-D feature\nvector per cell")
box(6.55, 2.6, 2.0, 1.0, "Rule-based screen\n(ALL / sickle / anemia)")

arrow(0.95, 4.3, 0.95, 3.6)
arrow(1.7, 3.1, 2.05, 3.1)
arrow(4.1, 3.1, 4.45, 3.1)
arrow(6.0, 3.1, 6.55, 3.1)

# --- Report box, spanning the width under the feature/rule row ---
box(2.05, 0.4, 6.5, 1.1, "Interactive report\ncell counts, WBC subtypes,\nscreening flags with evidence", fs=8.2, lw=1.3)

# feature vector -> report (straight down, no obstruction)
arrow(5.225, 2.6, 5.225, 1.5)
# rule-based screen -> report (straight down, no obstruction)
arrow(7.55, 2.6, 7.55, 1.5)
# stage2 / stage3 -> report: routed around the right margin to avoid crossing other boxes
elbow([(8.55, 5.21), (9.7, 5.21), (9.7, 0.95), (8.55, 0.95)])
elbow([(8.55, 4.16), (9.4, 4.16), (9.4, 1.15), (8.55, 1.15)])

plt.tight_layout(pad=0.3)
plt.savefig("paper/figs/fig1_cascade.png", dpi=280)
print("saved")
