# Review assessment and manuscript changes

The supplied review was checked against the implemented method and saved experiment records. This revision changes presentation and reporting of completed work. It does not retrain models, change the test partition, or add the experiments proposed by the reviewer.

The contribution remains a supervised ALL evidence regularizer and its controlled evaluation. The results do not establish improved accuracy, biological localization, clinical validity or overall robustness.

| Review point | Evidence and revision |
| --- | --- |
| Meaning of the intervention images | `src/evidence_regularization.py` erases the stain proposal in one view and an equal number of context pixels in the other. Section IV now names both views explicitly. The context erased view retains the stain proposal; neither view isolates all diagnostic information. |
| Interaction between classification and ranking | `tools/run_evidence_study.py` averages the original and context erased classification losses. The explanation now states that this preserves supervised weight per source. The class margin already includes all rival logits. No additional rival class or contrastive objective was tested. |
| Frozen backbone | The feature extractor is fixed and only the linear head is fitted. The manuscript retains the rationale of isolating the training objective and enabling CPU reproduction, and now explicitly notes that spatial features cannot adapt. Fine tuning remains untested. |
| Attribution diagnostics | No saved study results establish saliency overlap, deletion or insertion performance, or a shift of attribution toward the stain proposal. Section VI now states this directly. Predictive comparisons are not presented as attribution validation. |
| Mask design and normalization | The experiment uses one saturation proposal, fixed morphology, a half-image cap, deterministic context selection and grey fill 127. Inpainting, random masks, alternative area caps, stronger proposals and stain normalization are now explicitly identified as untested. |
| Probability and calibration reporting | The existing `evaluation.none` entries in `outputs/all_novel/study.json` contain NLL after validation-fitted temperature scaling. Table V now reports those recorded values for all five families. The saved values are checked against the saved probabilities. Expected calibration error, Brier scores and comparisons before and after scaling were not evaluated in this study and are not invented. |
| Class-specific errors | The existing AMER confusion matrix contains 8 Early, 3 Pre, 2 Benign and 0 Pro errors. The results now point out the concentration of 8 of 13 errors in Early images; Figure 3 retains all counts. |
| Alternative localization and region aggregation methods | No multiple instance learning, cell-region aggregation, attention baseline or stronger region model was trained for this comparison. The paper identifies the absence of a region aggregation baseline. The review's suggested methods are not described as completed work or appended as unverified comparisons. |
| Grouping reproducibility | The project includes source group and split assignments, pixel and file hashes in `data/ALL_verified/manifest.csv`, grouping audit records, model hashes, selection records and deterministic mask code. Section IV now describes these records. Original images are required to reconstruct masks; masks are not expert annotations. |
| External validation and detection on ALL | Patient identifiers and labelled ALL detection boxes remain unavailable. The existing qualitative detector observation is retained. No quantitative ALL detection result, combined clinical accuracy or external patient evaluation is claimed. |

The table titles now use regular Times letterforms in sentence case instead of small capitals. Section headings also use regular letterforms. Table borders, short text hyphens, two authors per row, six pages, 31 references and omission of the acknowledgment are retained.

The study file remains unchanged with SHA-256:
`1540495e7f02f29b3d115d0975d25f15673282d8ee113728af589e48819d8bfb`.

The current paper and source packages replace the previous delivery under the same filenames. This assessment records what was addressed and what requires further research; it does not claim that every review recommendation has been completed.
