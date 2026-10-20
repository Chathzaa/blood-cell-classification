# Prospective development protocol: area-matched evidence regularization

Scope remains blood cell detection and image-level acute lymphoblastic leukemia (ALL) screening. No anemia or sickle cell component is added.

## Research question and proposed distinction

Can a classifier be trained to depend more on stained regions than on equally large context regions, while preserving whole-image screening accuracy? The candidate method is area-matched evidence regularization (AMER). It compares the true-category margin after erasing a stain-region proposal with the margin after erasing the same number of surrounding pixels. A ranking loss favours stronger evidence in the stain regions. It supplements, rather than renames, an ordinary classification loss and context-erasure augmentation. The area-matched comparison is the specific proposed mechanism. Stain proposals are not expert cell or nucleus masks. No new general theory, first-ever claim, or clinical causality is asserted.

Relevant prior work already covers counterfactual/invariant image augmentation (Chang et al., CVPR 2021), biomarker-sensitive ALL pipelines (Askari Farsangi et al., arXiv:2307.04014), and original/segmented ALL comparisons (Abdallat et al., Sensors 2026). The experiment must compare AMER with ordinary context augmentation and a ranking ablation without area matching, in addition to the original-view head. A literature search is not proof of worldwide novelty.

## Data status

Use the existing 3,216-image deduplicated manifest and source groups. Train and validation remain the only fitting and selection partitions. The previous 643-image test results have already been inspected during the prior revision. Therefore this extension is exploratory evaluation on a previously used test partition, not an untouched confirmatory test. Do not resplit to conceal this fact. Patient identifiers and external clinical validation remain absent. Preserve every prior experiment.

Before fitting, examine a fixed first-four-per-category training-image detector probe. It shows poor transfer of the BCCD WBC detector to some ALL fields. Do not treat detector output as expert leukemia localization. Detection remains a separately evaluated branch.

## Image intervention fixed before fitting

Resize to 224 by 224. Estimate a stain proposal by Otsu thresholding HSV saturation, close with a 3 by 3 kernel, and dilate with a 5 by 5 elliptical kernel. If the proposal exceeds half the pixels, use the most saturated half; if empty, mark the intervention unavailable and use the ordinary classification loss only. Equal-area context consists of the same number of nonproposal pixels furthest from the proposal (distance transform, stable raster tie breaks). For the unmatched control, all nonproposal pixels are erased. Erasure fills selected pixels with neutral grey 127 in all channels. Original class labels are used for original and context-erased inputs only; no benign label is invented for stain-erased inputs.

## Frozen representation, training and selection

Use the same frozen ImageNet YOLOv8 nano backbone and 1,280 features as the previous study. Fit standardization on original training features only. Each candidate is a four-category linear head with class-balanced cross entropy and L2 weight decay. Baseline families: original only; original plus area-matched context erasure; original plus full-context erasure; AMER; AMER without area matching. AMER adds softplus(1 - margin(context-erased) + margin(stain-erased)), where margin is the true logit minus logsumexp of the other three logits. The erasure pair is used only for valid masks. Average the original and augmented classification losses, so source images keep equal total supervised weight.

For all families, L2 strengths are 0.0001, 0.001 and 0.01. For the two ranking families, ranking weights are 0.1 and 1.0. Optimize convex original/augmentation heads and the ranking heads with deterministic full-batch L-BFGS, initialized at zero, fixed maximum 400 iterations. Track convergence. Select within each family by clean validation macro F1, then validation negative log likelihood, then stronger regularization. Temperature calibration uses clean validation only. Save all candidate validation results and selected artifact hashes before test evaluation. Do not select a winning family from the test set: the proposed family is AMER regardless of its result.

## Fixed evaluation

Evaluate all five families on the same 643 images: original, the prior Gaussian blur (5 by 5, sigma 1), the prior whole-image BGR gains (0.9,1,1.1), and a new context-only colour perturbation with BGR gains (0.75,1,1.25) outside the original stain proposal. Report four-category accuracy, macro F1, binary sensitivity/specificity, and confusion counts. Also report how often the prediction changes under context perturbation. Context perturbation is a synthetic experiment, not another hospital or patient population. Use paired group bootstrap confidence intervals for differences against original-only and ordinary matched-context augmentation. Report all conditions, including negative findings. Inference uses only the original image and a single selected head; intervention images are training tools and an optional explanation audit.

## Decision criteria

A positive result requires evidence beyond a renamed combination: a measured benefit relative to ordinary matched augmentation, with explicit uncertainty and tradeoffs. A nonpositive result is retained and the method described as a research candidate, not proven superior. New experiments cannot retroactively restore independence of the reused test set. The paper must disclose this limitation prominently.

## Numerical amendment before test evaluation

The 400-iteration run converged for baseline families but reached the cap for all AMER candidates. Before any new test evaluation, the maximum was increased uniformly to 2,000 iterations and all five families were refitted. Hyperparameter grids, validation selection, masks and evaluation conditions are unchanged. The initial converged baseline artifacts are preserved in runs/all_novel/numerical_attempt_400. Feature cache provenance separates image extraction from the optimizer implementation.
