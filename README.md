# Blood Cell Detection and Acute Lymphoblastic Leukemia Screening

University of Ruhuna, Department of Electrical and Information Engineering

The current six-page IEEE conference paper is [main.pdf](paper/final_conference/main.pdf). Its [LaTeX source](paper/final_conference/main.tex), figures and [build instructions](paper/final_conference/BUILD.md) are kept together. This is the only manuscript version in the repository. The paper includes 31 references and fully bordered tables with regular Times titles. The [review assessment](paper/REVIEW_RESPONSE.md) distinguishes completed work from suggested future experiments.

## Contribution and measured results

The paper proposes area matched evidence regularization (AMER) for ALL image screening. Training compares removal of automatic stain regions with removal of the same number of context pixels. A supervised ranking loss encourages stronger support for the correct category when the stain regions remain visible. Five training variants test the effects of augmentation, ranking and area matching. This is a specific adaptation of existing evidence training research; the general principle is not claimed as an invention.

| Component | Evaluation | Result |
| --- | --- | --- |
| Blood cell detector | 72 BCCD fields, 945 objects | mAP at IoU 0.50: 0.910 |
| Normal WBC crop classifier | 63 original sources | 87.3% classification accuracy |
| ALL original image control | 643 test images | 98.8% accuracy across four categories |
| ALL AMER classifier | Same 643 images | 98.0% accuracy; macro F1 0.977 |
| ALL AMER binary screening | Same 643 images | Sensitivity 98.7%; specificity 97.9% |

AMER did not improve performance overall. The paper reports all five model families and all fixed perturbations, including negative findings. The ALL test partition was used in an earlier experiment, so this extension is exploratory. Patient identifiers are unavailable and image grouping does not establish patient independence. Joint clinical accuracy of cell detection and ALL screening has not been measured.

## Run the application

Use Python 3.12:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The repository includes the fitted models required for inference. Choose combined analysis or ALL screening and enable **Use experimental evidence training** to use the AMER model reported in the paper. The previous original/stain screening method remains the default because AMER did not demonstrate an overall improvement.

Other modes detect RBCs, WBCs and platelets, or classify a normal WBC crop. Combined ALL mode withholds normal WBC subtype assignments and does not assign the image's leukemia label to every detected cell. The system covers ALL image screening, not anemia, sickle cell disease, other leukemia types or patient diagnosis. Benign, Early, Pre and Pro are dataset categories, not a severity scale.

## Source audit and reproducibility

The original WBC validation split had augmentation relatives in training for all 2,476 validation images. The corrected cell study uses original source fields. The ALL audit removes 40 exact duplicates and groups observable relatives before partitioning 3,216 images into 1,928 training, 645 validation and 643 test images. Dataset source manifests and recorded predictions are included; original image collections and large feature caches are excluded from Git.

- [Cell protocol](audit/EXPERIMENT_PROTOCOL.md)
- [Original ALL protocol](audit/LEUKEMIA_PROTOCOL.md)
- [AMER protocol](audit/NOVEL_METHOD_PROTOCOL.md)
- [AMER training and evaluation](tools/run_evidence_study.py)
- [Recorded AMER comparisons](outputs/all_novel/study.json)
- [Source audit report](AUDIT_AND_REVISION_REPORT.md)

The fitting code, transformation code and protocols are bound to recorded hashes. Keep exact file bytes when reproducing the results. Full image audits and integration tests require the source datasets at the paths recorded in the manifests. Training also requires the source images; cached features can be regenerated with the study scripts. Reusing the same test set does not provide independent confirmation.

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest tests -q
python tools/verify_release.py
```

## Build and export the paper

Use XeLaTeX on `paper/final_conference/main.tex` until references settle, or use Tectonic. The source package contains all figures and generated tables; no dataset is needed to typeset it. For Overleaf, choose XeLaTeX.

```powershell
python paper/final_conference/make_assets.py
python tools/finalize_conference_release.py
```

Regenerating figures requires the original training image used in the intervention example. Packaging uses the current source and recorded evidence directly. The local `deliverables` directory contains the final PDF, LaTeX ZIP and project ZIP. Archives, compiler caches, intermediate TeX files and superseded manuscripts are not committed.
