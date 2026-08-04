"""
manual_confusion_matrix.py
==========================
Produces a TRUE 4x4 classification confusion matrix for the Stage 2 WBC
subtype model -- with NO 'background' row/column.

WHY THIS EXISTS
---------------
Ultralytics' model.val() builds an OBJECT DETECTION confusion matrix of size
(nc+1) x (nc+1). The extra row/column is 'background':

    background COLUMN = detections matching no ground-truth box -> FALSE POSITIVES
    background ROW    = ground-truth boxes matched by no detection -> FALSE NEGATIVES
    bottom-right cell = never populated by Ultralytics; structurally meaningless

The matrix is COLUMN-normalised (each column sums to 1).

So the "1.00" your supervisor flagged, sitting at
(row=Neutrophil, column=background), means:

    "Of all detections that matched nothing real, 100% were labelled Neutrophil."

It is a false-positive statistic, not an accuracy. It normalises to 1.00 simply
because that column has only one non-zero entry.

But Stage 2 is not really a detection problem at all. prepare_dataset.py writes
every label as:

    class_id 0.5 0.5 0.9 0.9

i.e. one object per image, always centred, always 90% of the frame. There is
nothing to localise. It is a 4-class CLASSIFICATION problem wearing a detection
costume -- which is exactly why the background row/column appears and why
mAP/precision/recall come out at ~99.5%.

This script evaluates it as what it is: pick the highest-confidence detection per
image, compare to the ground-truth class, and build a plain 4x4 matrix.

USAGE
-----
Detection model (current stage2 weights):

    python manual_confusion_matrix.py \
        --weights runs/detect/runs/stage2_classification/weights/best.pt \
        --data    data/BloodCell_YOLO \
        --split   val \
        --out     outputs/

Classification model (if/when you retrain with yolov8n-cls, recommended):

    python manual_confusion_matrix.py \
        --weights runs/classify/stage2_cls/weights/best.pt \
        --data    data/BloodCellImages/dataset2-master/images/TEST \
        --mode    cls \
        --out     outputs/

Requires: ultralytics, scikit-learn, matplotlib, seaborn, numpy
    pip install scikit-learn seaborn
"""

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")          # no display needed
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import (
        confusion_matrix,
        classification_report,
        accuracy_score,
        matthews_corrcoef,
    )
    from ultralytics import YOLO
except ImportError as e:
    sys.exit(
        f"Missing dependency: {e}\n"
        "Install with:  pip install ultralytics scikit-learn seaborn matplotlib"
    )


# Order MUST match the 'names' list in bloodcell.yaml
CLASS_NAMES = ["Eosinophil", "Lymphocyte", "Monocyte", "Neutrophil"]

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def collect_detection_split(data_root: Path, split: str):
    """
    YOLO detection layout:
        data_root/<split>/images/*.jpg
        data_root/<split>/labels/*.txt   (first token of first line = class id)

    Returns list of (image_path, true_class_id).
    """
    img_dir = data_root / split / "images"
    lbl_dir = data_root / split / "labels"

    if not img_dir.is_dir():
        sys.exit(f"Image directory not found: {img_dir}")
    if not lbl_dir.is_dir():
        sys.exit(f"Label directory not found: {lbl_dir}")

    samples = []
    skipped = 0

    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in IMG_EXTS:
            continue

        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            skipped += 1
            continue

        text = lbl_path.read_text().strip()
        if not text:
            skipped += 1          # empty label file -> no ground truth
            continue

        try:
            true_id = int(text.splitlines()[0].split()[0])
        except (ValueError, IndexError):
            skipped += 1
            continue

        if not 0 <= true_id < len(CLASS_NAMES):
            skipped += 1
            continue

        samples.append((img_path, true_id))

    if skipped:
        print(f"  [warn] skipped {skipped} images (missing/empty/malformed label)")

    return samples


def collect_classification_split(data_root: Path):
    """
    Classification layout (folder-per-class):
        data_root/EOSINOPHIL/*.jpeg
        data_root/LYMPHOCYTE/*.jpeg
        ...
    Folder names are matched case-insensitively against CLASS_NAMES.

    Returns list of (image_path, true_class_id).
    """
    if not data_root.is_dir():
        sys.exit(f"Data directory not found: {data_root}")

    lookup = {name.lower(): i for i, name in enumerate(CLASS_NAMES)}
    samples = []

    for sub in sorted(data_root.iterdir()):
        if not sub.is_dir():
            continue
        true_id = lookup.get(sub.name.lower())
        if true_id is None:
            print(f"  [warn] unrecognised class folder, skipping: {sub.name}")
            continue
        for img_path in sorted(sub.iterdir()):
            if img_path.suffix.lower() in IMG_EXTS:
                samples.append((img_path, true_id))

    return samples


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict(model, samples, mode, conf):
    """
    Run the model over every sample.

    Returns (y_true, y_pred, n_no_detection).

    Images where a DETECTION model finds nothing are EXCLUDED from the matrix
    and reported separately -- lumping them into a 5th 'background' class is
    precisely the artefact we are trying to remove.
    """
    y_true, y_pred = [], []
    no_detection = 0

    for i, (img_path, true_id) in enumerate(samples, 1):
        if i % 200 == 0 or i == len(samples):
            print(f"  {i}/{len(samples)}", end="\r", flush=True)

        results = model(str(img_path), verbose=False, conf=conf)
        r = results[0]

        if mode == "cls":
            # Classification head: probs.top1 is the predicted class index
            pred_id = int(r.probs.top1)
        else:
            # Detection head: take the single highest-confidence box.
            # Every Stage 2 image contains exactly one cell, so this is the
            # correct reduction from detection output to a class label.
            if r.boxes is None or len(r.boxes) == 0:
                no_detection += 1
                continue
            confs = r.boxes.conf.cpu().numpy()
            pred_id = int(r.boxes.cls.cpu().numpy()[int(np.argmax(confs))])

        y_true.append(true_id)
        y_pred.append(pred_id)

    print()
    return np.array(y_true), np.array(y_pred), no_detection


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_matrix(cm, normalise, title, out_path):
    if normalise:
        # Row-normalise: each row = one TRUE class, sums to 1.
        # Diagonal then reads as per-class RECALL, which is what people expect
        # from "normalised confusion matrix" in a classification paper.
        with np.errstate(divide="ignore", invalid="ignore"):
            data = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        data = np.nan_to_num(data)
        fmt, vmax = ".2f", 1.0
    else:
        data, fmt, vmax = cm, "d", None

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        data,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        vmin=0,
        vmax=vmax,
        square=True,
        cbar=True,
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        linewidths=0.5,
        linecolor="lightgrey",
    )
    plt.xlabel("Predicted label", fontsize=11)
    plt.ylabel("True label", fontsize=11)
    plt.title(title, fontsize=12, pad=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Corrected 4x4 classification confusion matrix for the "
                    "Stage 2 WBC subtype model (no background row/column)."
    )
    ap.add_argument("--weights", required=True, help="path to best.pt")
    ap.add_argument("--data", required=True,
                    help="detection: dataset root (contains <split>/images). "
                         "cls: folder-per-class directory.")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"],
                    help="detection mode only (default: val)")
    ap.add_argument("--mode", default="detect", choices=["detect", "cls"],
                    help="model type (default: detect)")
    ap.add_argument("--conf", type=float, default=0.25,
                    help="confidence threshold, detection mode (default: 0.25)")
    ap.add_argument("--out", default="outputs", help="output directory")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data)

    print("=" * 62)
    print("CORRECTED CONFUSION MATRIX  (4x4, no background row/column)")
    print("=" * 62)
    print(f"weights : {args.weights}")
    print(f"data    : {data_root}")
    print(f"mode    : {args.mode}"
          + (f"  |  split: {args.split}" if args.mode == "detect" else ""))
    print()

    # --- load ---------------------------------------------------------------
    model = YOLO(args.weights)
    print(f"Model class names: {model.names}")
    if len(model.names) != len(CLASS_NAMES):
        print(f"  [WARNING] model has {len(model.names)} classes but this script "
              f"expects {len(CLASS_NAMES)}: {CLASS_NAMES}")
        print("  If these weights are stage1 or stage3, you have the wrong file.")
    print()

    # --- collect ------------------------------------------------------------
    print("Collecting ground truth...")
    if args.mode == "cls":
        samples = collect_classification_split(data_root)
    else:
        samples = collect_detection_split(data_root, args.split)

    if not samples:
        sys.exit("No samples found. Check --data / --split / --mode.")
    print(f"  {len(samples)} images\n")

    # --- predict ------------------------------------------------------------
    print("Running inference...")
    y_true, y_pred, no_det = predict(model, samples, args.mode, args.conf)

    if no_det:
        pct = 100 * no_det / len(samples)
        print(f"\n  [note] {no_det} image(s) ({pct:.1f}%) produced NO detection "
              f"at conf={args.conf}.")
        print("         These are excluded from the matrix and reported here "
              "instead of being")
        print("         folded into a fake 'background' class.\n")

    if len(y_true) == 0:
        sys.exit("Model produced no predictions at all. Try lowering --conf.")

    # --- matrix -------------------------------------------------------------
    labels = list(range(len(CLASS_NAMES)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    print("=" * 62)
    print("CONFUSION MATRIX (raw counts)   rows = TRUE, cols = PREDICTED")
    print("=" * 62)
    header = " " * 13 + "".join(f"{n[:9]:>11}" for n in CLASS_NAMES)
    print(header)
    for i, name in enumerate(CLASS_NAMES):
        row = f"{name[:12]:<13}" + "".join(f"{v:>11d}" for v in cm[i])
        print(row)
    print()

    acc = accuracy_score(y_true, y_pred)
    # MCC uses every cell of the confusion matrix and remains informative when
    # classes are imbalanced.  It is therefore reported alongside accuracy.
    mcc = matthews_corrcoef(y_true, y_pred)
    print(f"Overall accuracy: {acc:.4f}  ({acc * 100:.2f}%)")
    print(f"Matthews correlation coefficient (MCC): {mcc:.4f}")
    print(f"Evaluated on {len(y_true)} images"
          + (f" ({no_det} excluded: no detection)" if no_det else ""))
    print()

    print("=" * 62)
    print("PER-CLASS METRICS")
    print("=" * 62)
    report = classification_report(
        y_true, y_pred,
        labels=labels,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )
    print(report)

    # --- save ---------------------------------------------------------------
    plot_matrix(cm, False,
                "Confusion Matrix (counts)",
                out_dir / "confusion_matrix_counts.png")
    plot_matrix(cm, True,
                "Confusion Matrix (row-normalised; diagonal = recall)",
                out_dir / "confusion_matrix_normalized.png")

    txt = out_dir / "classification_report.txt"
    with open(txt, "w") as f:
        f.write("CORRECTED CONFUSION MATRIX -- Stage 2 WBC subtype\n")
        f.write("=" * 62 + "\n")
        f.write(f"weights : {args.weights}\n")
        f.write(f"data    : {data_root}\n")
        f.write(f"mode    : {args.mode}\n")
        if args.mode == "detect":
            f.write(f"split   : {args.split}\n")
            f.write(f"conf    : {args.conf}\n")
        f.write(f"\nEvaluated : {len(y_true)} images\n")
        f.write(f"No detection (excluded): {no_det}\n")
        f.write(f"\nOverall accuracy: {acc:.4f}\n\n")
        f.write(f"Matthews correlation coefficient (MCC): {mcc:.4f}\n\n")
        f.write("Confusion matrix (rows=TRUE, cols=PREDICTED):\n")
        f.write(header + "\n")
        for i, name in enumerate(CLASS_NAMES):
            f.write(f"{name[:12]:<13}" + "".join(f"{v:>11d}" for v in cm[i]) + "\n")
        f.write("\n" + report + "\n")
    print(f"  saved: {txt}")

    print()
    print("=" * 62)
    print("Use confusion_matrix_normalized.png to replace slide 13.")
    print("It is a genuine 4x4 classification matrix. No background row/column,")
    print("because this was never a detection problem to begin with.")
    print("=" * 62)


if __name__ == "__main__":
    main()
