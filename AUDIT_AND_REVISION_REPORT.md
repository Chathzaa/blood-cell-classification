# Blood cell detection and leukemia screening revision

The user clarified the required scope: blood cell detection plus leukemia screening, without anemia or sickle cell predictions. This release restores that scope with a new ALL experiment, rather than treating the old leukemia model's validation score as independent evidence.

## What the system now does

- Detects RBCs, WBCs and platelets and preserves one object record per detection.
- Supports normal WBC subtype analysis using the corrected crop-trained model.
- Screens complete images for benign-like or ALL-like patterns using two newly fitted heads.
- Reports an ALL dataset subtype only when both views support it; otherwise retains the broader ALL flag or requests full review.
- Offers a combined report while withholding normal WBC subtype labels on potentially malignant images.

The ALL label belongs to the whole image, not each detected WBC. The cell detector and ALL classifier are evaluated on their respective labelled collections. Joint clinical accuracy is not established.

## Why the previous leukemia score was insufficient

The old checkpoint achieved 640/641 on a validation set used for checkpoint selection. Exact deduplication removed 40 redundant files, but further visually related fields remained across its split. Patient-to-image identifiers are absent. The four labels describe dataset categories, not ordered severity. These issues did not require abandoning leukemia screening; they required a new experiment and narrower claims.

## New ALL study

The 3,256 originals become 3,216 unique decoded-pixel images. Conservative perceptual grouping and affine registration find 70 pair links, producing 3,147 groups, including 68 groups with multiple images. Groups never cross the split. Representative accepted pairs were visually inspected, and no group has conflicting labels.

| Partition | Images | Image groups |
| --- | ---: | ---: |
| Training | 1,928 | 1,888 |
| Validation | 645 | 630 |
| Test | 643 | 629 |

These are image groups, not patient groups. The original and stain-focused views always inherit the same group.

The new models start from the supplied ImageNet backbone, not a supplied leukemia model. Frozen 1,280-dimensional features feed separate class-balanced multinomial logistic heads. Standardization is fitted on training data. Validation selects C, temperature and review thresholds. Both heads converge well below their iteration limit. The supplied Segmented images are not used for fitting.

The protocol was written before fitting and new test prediction. It also fixed blur and colour perturbations, the confidence-only comparator, and the rule for retaining a broad ALL flag while withholding an uncertain subtype.

## Results and novelty assessment

| ALL model | Four-category accuracy | Macro F1 | Binary sensitivity | Binary specificity |
| --- | ---: | ---: | ---: | ---: |
| Original view | 98.8% | 0.986 | 99.3% | 99.0% |
| Stain-focused view | 94.9% | 0.939 | 97.8% | 84.4% |
| Mean fusion | 98.6% | 0.985 | 99.5% | 99.0% |

The consistency rule accepts 611 binary decisions with no observed error and defers 32. It reports 526 malignant subtypes, with two subtype errors, and preserves five ALL flags with subtype review. Benign cases are deferred more often than malignant cases, so coverage is reported by class.

At the same clean-test coverage, confidence-only fusion also has zero binary errors. Under blur, consistency retains seven errors versus three for confidence alone; under colour gain, it retains eleven versus nine. Fusion modestly improves four-category accuracy over the original head under those perturbations, but agreement is not the better rejection rule in this experiment.

The defensible contribution is the source-audited, controlled evaluation of view-consistent hierarchical screening within a cell-analysis workbench, including these limitations. It is not a claim to have invented YOLO, two-view learning or selective classification. A particular conference may require stronger algorithmic novelty; no acceptance or first-ever claim is made.

## Cell corrections retained

The earlier audit found that all 2,476 legacy WBC validation images had augmentation relatives in training. The original preparation extracted a random suffix as a source ID and lost eight colliding filenames. The corrected data inherit original BCCD partitions, exclude the related validation field 00037, and use terminal epoch 60 to avoid selecting the new classifier on contaminated nominal validation.

The primary WBC crop classifier has 87.3% accuracy on 63 annotated crops; the detector/classifier cascade has 85.7% accuracy and macro F1 0.845, including three extra WBC predictions. The frozen detector has mAP 0.910 at IoU 0.50 on 72 fields. Its historical checkpoint selection used a related validation scene; the limitation remains disclosed.

Counting no longer duplicates WBC subtypes as extra objects. Missing contours no longer suppress classification or invent morphology. Stain and texture measurements respect their masks. The obsolete disease heuristics, CBC assertions, unmeasured throughput and segmentation Dice claims remain removed.

## Verification and delivery

Nineteen regression tests pass. Live ALL predictions match stored probabilities, all four app modes start without exceptions, and a real combined analysis completes while withholding normal subtype labels. Registry hashes protect the evaluated checkpoints and transformations.

The current manuscript is paper/final_conference/main.tex and main.pdf. It has six IEEE conference pages, 31 references and four authors arranged two per row. It reports the AMER formulation and all five controlled comparisons. Figures and tables come from recorded evidence. All six tables have complete borders and cell dividers; titles use regular letterforms, text uses short hyphens and the acknowledgment is omitted. The review revision adds already-recorded NLL to Table V, clarifies the loss and intervention views, and identifies unperformed analyses as limitations. See paper/REVIEW_RESPONSE.md. Earlier manuscript versions have been removed; this report retains the preceding experiment history.

The earlier WBC-focused deliverables are preserved under deliverables/previous_wbc_focus. Original code remains in audit/original_source, with the immediately preceding app and documentation also preserved under audit/before_leukemia_revision.

## Remaining limits

No patient identifiers, external clinical cohort or prospective outcomes are available. Similarity grouping can miss changed views. Normal WBC categories are not validated labels for malignant cells. Descriptive masks have no expert segmentation benchmark. Controlled blur/colour tests are not external validation. ALL screening does not cover AML, CML, CLL or other leukemia types. The current results support a reproducible research prototype, not a clinical diagnostic device.


## Evidence-training extension, 12 October 2026

A proposed ALL-specific area matched evidence regularizer and four controls were implemented and evaluated. The original topic and default inference remain intact. The candidate did not establish a beneficial replacement: clean accuracy is 98.0% versus 98.8% for original-only training, and global-colour performance worsened. See NOVEL_METHOD_FINDINGS.md for all results, the closest prior art, test-reuse disclosure, and the unresolved novelty requirement. The optional candidate is available in the app; its paper is an experimental six-page draft with 31 references. Twenty-seven regression tests, six app startup variants, real combined inference and source/result verification passed.
