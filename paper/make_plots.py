import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
})

# ---------- 1. Clean 4x4 confusion matrices (no background row/col) ----------

def plot_cm(cm, labels, title, out_path):
    cm = np.array(cm, dtype=float)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = cm / row_sums

    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Predicted", fontsize=8.5)
    ax.set_ylabel("True", fontsize=8.5)
    ax.set_title(title, fontsize=9.5)

    for i in range(len(labels)):
        for j in range(len(labels)):
            val = cm_norm[i, j]
            txt = f"{val:.2f}\n({int(cm[i,j])})"
            color = "white" if val > 0.5 else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7, color=color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=260)
    plt.close(fig)

stage2_labels = ["Eosinophil", "Lymphocyte", "Monocyte", "Neutrophil"]
stage2_cm = [[626, 0, 0, 0], [0, 620, 0, 0], [0, 0, 614, 0], [0, 0, 0, 616]]
plot_cm(stage2_cm, stage2_labels, "Stage 2: WBC subtype", "paper/figs/fig3a_stage2_confusion_matrix.png")

stage3_labels = ["Benign", "Early", "Pre", "Pro"]
stage3_cm = [[99, 0, 0, 0], [1, 197, 0, 0], [0, 0, 192, 0], [0, 0, 0, 152]]
plot_cm(stage3_cm, stage3_labels, "Stage 3: ALL severity", "paper/figs/fig3b_stage3_confusion_matrix.png")

# ---------- 2. Stage 1 per-class metric bar chart ----------

classes = ["RBC", "WBC", "Platelet"]
precision = [0.804, 0.930, 0.861]
recall = [0.752, 1.000, 0.899]
map50 = [0.864, 0.962, 0.903]

x = np.arange(len(classes))
w = 0.25
fig, ax = plt.subplots(figsize=(3.6, 2.8))
ax.bar(x - w, precision, width=w, label="Precision", color="#4C72B0")
ax.bar(x, recall, width=w, label="Recall", color="#DD8452")
ax.bar(x + w, map50, width=w, label="mAP@0.50", color="#55A868")
ax.set_xticks(x)
ax.set_xticklabels(classes)
ax.set_ylim(0, 1.08)
ax.set_ylabel("Score")
ax.set_title("Stage 1 detection: per-class metrics", fontsize=9.5)
ax.legend(fontsize=7, loc="lower right", framealpha=0.9)
ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.6)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
plt.tight_layout()
plt.savefig("paper/figs/fig5_stage1_perclass_bars.png", dpi=260)
plt.close(fig)

# ---------- 3. Training curves for Stage 2 and Stage 3 (real results.csv) ----------

def load_curve(path):
    rows = list(csv.DictReader(open(path)))
    epoch = [int(r["epoch"]) for r in rows]
    acc = [float(r["metrics/accuracy_top1"]) for r in rows]
    loss = [float(r["train/loss"]) for r in rows]
    return epoch, acc, loss

e2, a2, l2 = load_curve("runs/classify/runs/stage2_cls_groupsplit/results.csv")
e3, a3, l3 = load_curve("runs/classify/runs/stage3_cls_dedup/results.csv")

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))

ax = axes[0]
ax.plot(e2, a2, color="#4C72B0", linewidth=1.3, label="Stage 2 (WBC subtype)")
ax.plot(e3, a3, color="#C44E52", linewidth=1.3, label="Stage 3 (ALL severity)")
ax.set_xlabel("Epoch")
ax.set_ylabel("Val top-1 accuracy")
ax.set_ylim(0, 1.05)
ax.set_title("Validation accuracy vs. epoch")
ax.legend(fontsize=7, loc="lower right")
ax.grid(linestyle=":", linewidth=0.6, alpha=0.6)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

ax = axes[1]
ax.plot(e2, l2, color="#4C72B0", linewidth=1.3, label="Stage 2 (WBC subtype)")
ax.plot(e3, l3, color="#C44E52", linewidth=1.3, label="Stage 3 (ALL severity)")
ax.set_xlabel("Epoch")
ax.set_ylabel("Train loss")
ax.set_title("Training loss vs. epoch")
ax.legend(fontsize=7, loc="upper right")
ax.grid(linestyle=":", linewidth=0.6, alpha=0.6)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

plt.tight_layout()
plt.savefig("paper/figs/fig6_training_curves.png", dpi=260)
plt.close(fig)

print("all plots saved")
