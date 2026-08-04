# An Interpretable Cascaded System for Blood-Cell Analysis and Hematological-Abnormality Screening

N. A. N. N. N. Arachchi (EG/2021/4408), A. W. M. L. M. Bandara (EG/2021/4426), H. M. S. Muthukumari (EG/2021/4685), and C. Y. Samarasinghe (EG/2021/4775)  
Department of Electrical and Information Engineering, University of Ruhuna, Galle, Sri Lanka

## Abstract

Microscopic examination of peripheral-blood smears is essential for cell counting and the recognition of hematological abnormalities, but manual review is time-consuming and subject to observer variability. This paper presents an interpretable, browser-accessible prototype for blood-smear analysis. The system uses a three-stage cascade: YOLOv8 detection first localizes red blood cells (RBCs), white blood cells (WBCs), and platelets; a WBC-crop classifier assigns neutrophil, lymphocyte, monocyte, and eosinophil labels; and a second crop classifier screens WBC crops for leukemia-related classes. In parallel, contrast enhancement, marker-controlled watershed segmentation, and 22 morphological, colour, nuclear, and texture descriptors provide explainable measurements. Rule-based screening uses these measurements to flag patterns compatible with acute lymphoblastic leukemia (ALL), sickle-cell morphology, and anemia. The evaluation protocol reports class-wise precision, recall, F1-score, a standard 4 x 4 confusion matrix for WBC subtyping, and the Matthews correlation coefficient (MCC). MCC is included because it summarizes agreement over all confusion-matrix cells and is more robust than accuracy alone under class imbalance. Results are deliberately reported only after a group-aware held-out evaluation; placeholders in this draft must be replaced with the values produced by the supplied evaluation script. The system is an educational decision-support prototype, not a diagnostic medical device.

**Keywords—** blood smear analysis, YOLOv8, white blood cell classification, watershed segmentation, Matthews correlation coefficient, interpretable computer vision

## 1. Introduction

Blood-cell morphology and differential counts provide clinically useful evidence for hematological disorders. However, manual microscopic examination requires expertise and can be slow when many fields of view must be reviewed. Computer vision can assist by localizing cells, assigning cell types, measuring morphology, and presenting the evidence in a reproducible form.

This work develops a cascaded system rather than claiming a single model detects every cell type and disease state. Stage 1 detects RBCs, WBCs, and platelets in a full smear image. Stage 2 classifies each detected WBC crop into four subtypes. Stage 3 screens WBC crops using a separate leukemia classifier. Classical image processing supplies interpretable cell measurements, while transparent rules generate screening flags. This separation matches the available data: full-smear data are suitable for localization, whereas the subtype and leukemia data consist of individual-cell images.

The contributions are:

- a working cascade that combines full-smear detection, WBC subtype classification, leukemia screening, and an interactive reporting interface;
- watershed-based cell separation and a 22-feature morphology representation for explainable analysis;
- a verification protocol with a true 4 x 4 subtype confusion matrix, precision, recall, F1-score, and MCC; and
- an explicit account of dataset, validation, and clinical limitations.

## 2. Related Work

Deep detectors have been applied to blood-cell localization and counting, while cell-crop classifiers have been used for WBC subtyping. YOLOv8 offers a compact detector suitable for real-time prototype systems [1]. Classical preprocessing remains useful in stained-smear images: CLAHE improves local contrast [2], and marker-controlled watershed can divide touching objects [3]. Morphological measures such as area, circularity, eccentricity, and nucleus-to-cell ratio offer an interpretable complement to neural predictions.

Reported accuracy alone is insufficient when classes have unequal prevalence. A confusion matrix reveals which classes are confused, and MCC measures the correlation between predictions and labels using all cells of that matrix [4]. For multi-class classification, MCC is bounded by -1 and 1: 1 denotes perfect agreement, 0 performance no better than chance, and negative values systematic disagreement. Accordingly, this study treats MCC and per-class metrics as required verification outputs, not optional figures.

## 3. Materials and Methods

### 3.1 Data and split policy

Stage 1 uses the BCCD blood-cell detection dataset, whose annotations contain full-smear image bounding boxes. Stage 2 uses four single-cell WBC categories: eosinophil, lymphocyte, monocyte, and neutrophil. Stage 3 uses a public leukemia cell-image dataset containing benign and leukemia-related classes. The final manuscript must state the exact dataset versions, licence terms, image counts, and train/validation/test counts.

All evaluation sets must be held out before training. When augmented images share a common source image, every augmentation of that source must remain in only one split. This group-aware split prevents near-duplicate leakage and avoids inflated results.

### 3.2 Processing and detection cascade

Input images are read as 8-bit, three-channel BGR images. The preprocessing branch converts them to LAB colour space, applies CLAHE to the luminance channel, and denoises them. A binary Otsu mask undergoes morphological opening and closing; distance transform, marker construction, and watershed then separate touching regions. Watershed regions, rather than the unseparated binary mask, are used to obtain contours.

The Stage 1 YOLOv8 detector receives the original image because it was trained on original BCCD images. This prevents a training-inference distribution mismatch. Detected WBC boxes are padded and cropped. Stage 2 receives each crop and outputs one of four subtype labels. Stage 3 receives the same WBC crops, rather than a whole smear, because its training images contain individual cells.

### 3.3 Interpretable feature extraction and screening rules

For each contour, five geometric features are computed: area, perimeter, circularity, eccentricity, and convexity. Nuclear descriptors are nucleus-to-cell ratio and nuclear irregularity. HSV mean and standard deviation provide six colour features. Texture is described by GLCM energy, contrast, homogeneity, entropy, and correlation, together with a four-bin local-binary-pattern histogram. The feature vector has 22 values.

The screening layer is rule-based and produces a flag, confidence-like heuristic score, and textual evidence. ALL is flagged when blast percentage and abnormal-nucleus criteria are met; a lower-threshold rule produces a suspected flag. Sickle-cell screening requires both a high fraction of low-circularity RBCs and a high fraction of high-eccentricity RBCs. Anemia screening uses RBC size variation and pale-cell indicators. These are engineering screening heuristics, not clinically validated diagnostic thresholds.

### 3.4 Verification protocol and MCC

Stage 2 is evaluated as classification, not object detection: each crop has one true class. Let \(C\) be the four-class confusion matrix, \(s=\sum_{ij}C_{ij}\), \(c=\sum_k C_{kk}\), \(p_k=\sum_i C_{ik}\), and \(t_k=\sum_j C_{kj}\). Multi-class MCC is

\[
MCC=\frac{cs-\sum_k p_k t_k}{\sqrt{(s^2-\sum_k p_k^2)(s^2-\sum_k t_k^2)}}.
\]

The score is reported with accuracy, macro and weighted precision/recall/F1, raw counts, and a row-normalized 4 x 4 confusion matrix. No background row or column is included because background is an artefact of an object-detection confusion matrix and has no meaning for single-cell classification. Detection-stage evaluation separately reports mAP at specified IoU thresholds.

## 4. Results

### 4.1 Evaluation results

**Replace all bracketed values only after running the experiment. Do not use the older README claims unless the metrics can be reproduced.**

| Task | Held-out samples | Accuracy | Macro F1 | Weighted F1 | MCC |
|---|---:|---:|---:|---:|---:|
| Stage 2 WBC subtype classification | [N] | [ ] | [ ] | [ ] | [ ] |
| Stage 3 leukemia-crop classification | [N] | [ ] | [ ] | [ ] | [ ] |

For Stage 1 detection, report: mAP@0.50 = **[ ]**, mAP@0.50:0.95 = **[ ]**, precision = **[ ]**, and recall = **[ ]**, evaluated on **[N]** held-out full-smear images.

**Figure 1.** End-to-end cascade: raw smear -> Stage 1 detection -> WBC crops -> Stage 2 subtype / Stage 3 leukemia screening; preprocessing and watershed -> features -> rule-based screening.

**Figure 2.** Before-and-after watershed segmentation for a smear region containing touching RBCs.

**Figure 3.** Stage-2 row-normalized 4 x 4 confusion matrix. Insert `outputs/confusion_matrix_normalized.png` generated by `manual_confusion_matrix.py`.

### 4.2 Interpretation

The confusion matrix should be interpreted class by class. Its diagonal entries are recalls after row normalization; off-diagonal entries identify subtype confusions. MCC should be discussed alongside accuracy. A high accuracy accompanied by low MCC can signal an imbalanced or uninformative classifier, whereas a consistently high MCC supports agreement across classes. The result remains limited to the chosen held-out split and must not be interpreted as clinical diagnostic accuracy.

## 5. Discussion and Limitations

The pipeline provides useful traceability: detections, crops, subtype predictions, cell measurements, and rule evidence can all be inspected. Its cascade also avoids pretending that the same dataset or model solves detection, subtype classification, and leukemia screening equally well.

Several limitations are important. First, the data are public, 8-bit images and may not represent scanner bit depth, staining protocols, devices, magnification, or the patient population found in clinical practice. Second, morphology thresholds and disorder scores are heuristic and require expert-labelled clinical validation. Third, a crop-level leukemia classifier may face severe domain shift when applied to peripheral-smear WBC crops. Fourth, segmentation quality cannot be reported as Dice without independent ground-truth masks. Finally, neither the UI nor the screening logic is a medical device or a substitute for hematologist review.

Future work should obtain expert-annotated masks and disease labels, use patient- or source-group-independent splits, report calibration and uncertainty, compare the cascade to baselines, and conduct multi-site validation.

## 6. Conclusion

This paper presents an interpretable blood-smear analysis prototype that combines YOLOv8 detection, WBC-crop classification, watershed segmentation, morphology features, and rule-based screening. The proposed verification layer uses a genuine four-class confusion matrix and MCC in addition to accuracy and F1-score. The project should be presented as a reproducible educational decision-support prototype. Its future value depends on leakage-resistant evaluation and clinically supervised validation rather than unsupported headline accuracy claims.

## Declarations

**Funding:** No external funding declared. Update if applicable.  
**Conflict of interest:** The authors declare no conflict of interest.  
**Ethics and intended use:** This project analyses public image data. It is not a diagnostic device and must not be used to make clinical decisions without qualified review.

## References

[1] G. Jocher, A. Chaurasia, and J. Qiu, *Ultralytics YOLO*, 2023. [Online]. Available: https://docs.ultralytics.com/

[2] K. Zuiderveld, “Contrast limited adaptive histogram equalization,” in *Graphics Gems IV*, Academic Press, 1994, pp. 474–485.

[3] F. Meyer, “Topographic distance and watershed lines,” *Signal Processing*, vol. 38, no. 1, pp. 113–125, 1994.

[4] B. W. Matthews, “Comparison of the predicted and observed secondary structure of T4 phage lysozyme,” *Biochimica et Biophysica Acta*, vol. 405, no. 2, pp. 442–451, 1975.

[5] T. L. Bock et al., “A review of automated blood cell classification using deep learning,” [add the exact peer-reviewed source used by your team].

## Reproducibility checklist

1. Run the Stage-2 group-aware classifier evaluation: `python manual_confusion_matrix.py --weights <stage2-best.pt> --data <held-out-class-folders> --mode cls --out outputs/`.
2. Copy accuracy, class-wise precision/recall/F1, MCC, and the two confusion-matrix figures into Section 4.
3. Run Stage-1 validation on a genuinely held-out BCCD split and enter its mAP values.
4. Replace every bracketed placeholder, add final dataset counts/licences, then convert this Markdown draft to the IEEE two-column template.
