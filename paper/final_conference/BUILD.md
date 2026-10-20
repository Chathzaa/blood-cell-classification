# Final conference manuscript

**Blood Cell Detection and Leukemia Screening with Area Matched Evidence Regularization**

This package contains the six-page manuscript, 31 numbered references, generated results tables and three vector figures. The main source is `main.tex`. The contribution paragraph is the final paragraph of Section I, Introduction, on page 1.

Choose **XeLaTeX**, with `main.tex` as the main document, and compile until the reference warnings disappear (normally two or three passes):

```
xelatex -interaction=nonstopmode -halt-on-error main.tex
xelatex -interaction=nonstopmode -halt-on-error main.tex
```

For Overleaf, upload the ZIP as a new project and select XeLaTeX in the compiler settings. Python and the training datasets are not needed for typesetting. Keep the included `figures` directory and all `.tex` files together. The standard IEEEtran class is supplied by MiKTeX, TeX Live and Overleaf.

The verified PDF uses Times New Roman. Where that font is unavailable, the source falls back to TeX Gyre Termes; a font substitution can affect page breaks. The authors are arranged two per row. No email addresses have been invented.

The paper covers RBC, WBC and platelet detection, optional normal WBC classification, and image-level acute lymphoblastic leukemia screening. It proposes a supervised formulation of area matched evidence regularization and reports all five controlled training variants, including negative findings. The contribution is stated in relation to earlier evidence-training research, without claiming that the underlying general principle is new.

The evaluation remains exploratory: the ALL test partition was used in an earlier experiment, patient identifiers are unavailable, and joint cell detection and ALL diagnostic accuracy has not been established. These limitations are disclosed in the manuscript. No anemia or sickle cell diagnosis is included.

The layout uses the standard IEEE conference format because no specific conference template was supplied. Venue-specific anonymity, copyright and submission settings can be applied once the conference is selected.

In the full project, `python paper/final_conference/make_assets.py` regenerates figures and tables from recorded results. The paper reports `outputs/all_novel/study.json`; `tools/run_evidence_study.py` and `src/evidence_regularization.py` contain the implemented method. The application exposes AMER through its experimental evidence-training option and retains the earlier screening head as its default.

A portable Tectonic build is also supported: `tectonic --keep-logs main.tex`. Tables have complete outer borders and cell dividers. The template's small capitals are replaced by regular letterforms in table titles and section headings. Earlier manuscript versions are not required by the build or asset generator.

The review revision reports NLL already saved in the original study and clarifies unperformed analyses as limitations. It does not introduce new experiments or retrain models.
