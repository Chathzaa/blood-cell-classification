# A concrete research extension within the original topic

The topic remains **blood cell detection and acute lymphoblastic leukemia (ALL) screening**. Anemia and sickle cell disease are not added.

## Proposed contribution in simple English

During training, hide a stained region and measure the model's category support. Then hide an equally large area of surrounding context instead. Train the model to retain stronger support when the stained region is still visible. This is **area matched evidence regularization (AMER)**. At inference, the trained classifier needs only the original image; the existing blood cell detector remains available.

This is an ALL-specific methodological adaptation, not a new general theory. Explanation-guided learning, region intervention and matched evidence comparisons already exist. The closest papers are explicitly cited in audit/NOVELTY_LITERATURE_REVIEW.md and the draft. A new acronym or a different combination is not sufficient proof of strong novelty.

## What was actually built and tested

- A reproducible stain-proposal and equal-area context intervention.
- A four-category classifier with an explicit differentiable evidence-ranking loss.
- Five model families, trained on the same data and frozen representation: original-only, matched augmentation, full-context augmentation, AMER, and AMER without area matching.
- Validation-only model selection and calibration, saved before the extension's test predictions.
- Original images, blur, global colour changes, and context-only colour changes; all predictions and paired group bootstrap intervals are preserved.
- Optional experimental inference in the application. The earlier two-view screening model remains the default.

## Results: the candidate is not a proven improvement

Four-category accuracy on the same 643 images:

| Training method | Original | Blur | Global colour | Context colour |
|---|---:|---:|---:|---:|
| Original-only | 98.8% | 90.8% | 92.7% | 73.6% |
| Matched augmentation | 98.6% | 88.8% | 91.8% | 73.4% |
| Full-context augmentation | 98.1% | 93.3% | 92.2% | 78.2% |
| AMER | 98.0% | 89.7% | 86.8% | 74.2% |
| AMER without area matching | 97.7% | 93.2% | 89.7% | 75.0% |

AMER's small context-colour gain is uncertain: its paired 95% group interval against original-only training is -1.7 to +2.9 percentage points. Under global colour changes, its loss is 5.9 points, with a paired interval from -8.1 to -3.7 points. The ranking and area-matching controls therefore do not establish a beneficial replacement for the original-only classifier. No unsuccessful result was removed or retuned after evaluation.

The 643-image partition had already been evaluated in the earlier project revision. Although this extension uses only training and validation for fitting and selection, its test evaluation is **exploratory**, not fresh independent confirmation. Image groups are not patient identifiers.

## Other idea checked

A small training-image probe also examined using detected RBCs as an internal colour reference. At confidence 0.40, the first training image from each of the four ALL categories produced RBC counts 1, 0, 0, 0 at detector size 640, and 0, 0, 0, 0 at size 320. This is a limited feasibility probe, not a detection accuracy estimate. It provides insufficient support for an RBC-reference method with the present detector. The existing detector's BCCD accuracy does not transfer automatically to ALL fields.

## Paper and project status

The new six-page, 31-reference draft is **paper/evidence_revision/main.pdf**. Its contribution paragraph is the final paragraph of Section I on page 1, starting “We contribute an ALL specific evidence training scheme...”. Its abstract and results report the negative outcomes explicitly. The prior paper and all previous models are preserved.

This work supplies a concrete candidate method and a reproducible negative experiment. It does **not yet establish the strong novelty and beneficial evidence requested by the lecturer**, and should not be represented as a submission-ready solution to that requirement. A useful next research step needs a justified cell-region hypothesis, preferably expert region annotations, and an independent labelled evaluation collection. Adding more method names or hiding these comparisons would not fix that gap.

## Run or reproduce

Start the normal app with `python -m streamlit run app.py`. In a mode containing ALL screening, enable **Use experimental evidence training** to inspect AMER. The checkbox is off by default. Combined mode still detects blood cells and screens the whole image; it does not label every detected cell malignant.

The experiment is in `tools/run_evidence_study.py`. Model choices are in `runs/all_novel/selection.json`; results are in `outputs/all_novel/study.json`; the protocol and numerical amendment are in `audit/NOVEL_METHOD_PROTOCOL.md`. The 2,000-iteration run converged for all 21 candidates. Original data are required to reproduce feature extraction. Package verification is `python tools/verify_evidence_release.py`.

Validation completed: 27 regression tests, six app startup variants, real combined inference, and a six-page XeLaTeX build. The project archive includes the inference weights; original datasets and the virtual environment are not duplicated.
