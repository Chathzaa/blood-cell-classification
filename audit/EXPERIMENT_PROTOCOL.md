# Frozen evaluation protocol

Written before evaluation of any new classifier on the 63 original test images.

## Source controls

Use the BCCD source split for every representation of the same field. CSV subtype labels are accepted only for one supported category and one annotated white blood cell. All 339 corresponding source images in the detection and classification source collections have identical decoded pixels. No augmented legacy images or blood-cell classifier weights enter new model fitting.

The field-similarity audit runs before new test prediction. It proposes same-scene pairs using DCT hashes and SIFT affine registration. Held-out members of a scene shared with an earlier split are conservatively excluded from reported evaluation and review-threshold selection, and all accepted pairs are visually reviewed. No test image moves into training. Exact hashes and a similarity search are not evidence of patient independence.

## New classifiers

- YOLOv8 nano classification; identical supplied ImageNet initialization for every run.
- Two inputs: full image field or ground-truth WBC box with ten native pixels of context.
- 60 fixed epochs, AdamW, learning rate 0.001 with final multiplier 0.01, weight decay 0.0005, dropout 0.2, batch size 16, input size 224.
- Online horizontal and vertical reflection probability 0.5; automatic augmentation and random erasing disabled. No offline augmentation before splitting.
- Seeds 42, 43, 44. Seed 42 is the primary deployed run; additional seeds assess optimization variation on the same split.
- **Use the terminal `last.pt` checkpoint**, not the validation-selected `best.pt`.

The first runs began while the perceptual audit was still running. One nominal BCCD validation field was already suspected to share a scene with training. Selecting the terminal checkpoint avoids using contaminated validation accuracy to choose an epoch. Training runs the fixed 60 epochs without validation-based learning-rate scheduling or early stopping. The validation loop monitors performance but does not update gradients. Reported validation metrics and threshold selection use only the eligible evaluation manifest. This change was made before test evaluation. Historical train logs may still contain scores for the nominal validation split; these are not paper results.

## Evaluation

- Detector: supplied frozen three-class detector; inherited training provenance explicitly disclosed. Test AP is recomputed on its original 72-image test partition. Any related-scene test exclusions require an additional restricted detector evaluation.
- Oracle classification: each source's true WBC box (crop model) or full field (field model).
- Cascade: detector confidence 0.25, NMS IoU 0.70, one-to-one true/predicted WBC match at IoU >=0.50. Unmatched predictions count as false positives; unmatched true cells count as misses. An incorrectly subtyped matched cell contributes one false positive and one false negative for the appropriate classes.
- Report accuracy, macro F1, multiclass MCC for oracle classification, per-class errors, end-to-end macro F1 and accuracy, and Wilson 95% intervals for correctness proportions. Intervals describe image sampling, not independent patients.
- Review threshold: smallest threshold on 0.50,0.51,...,0.99 achieving at least 90% correctness among at least 20 matched validation cells. If impossible, review every cell. Report test coverage and error, without interpreting model scores as calibrated probabilities.
- Morphology baseline: train-only median imputation and standardization, balanced logistic regression, C selected from {0.01,0.1,1,10} using eligible validation macro F1. Test evaluated after C is frozen.
- A majority-class baseline accompanies the class-imbalanced test distribution.
- Legacy classifier numbers are repeated only as checkpoint-selection validation results, with their dependence problems explicitly stated.

No new disease, external-site, patient-level or segmentation accuracy is claimed.
