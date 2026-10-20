# Literature check for the revised contribution

Checked 11 October 2026. This is a focused search, not an exhaustive proof of novelty.

| Primary source | Existing contribution | Consequence for this project |
|---|---|---|
| Ross, Hughes and Doshi-Velez, *Right for the Right Reasons*, IJCAI 2017, https://arxiv.org/abs/1703.03717 | Regularizing classifiers using their input explanations. | Explanation-guided learning itself cannot be claimed as new. |
| Chang, Adam and Goldenberg, *Towards Robust Classification Model by Counterfactual and Invariant Data Generation*, CVPR 2021, https://arxiv.org/abs/2106.01127 | Perturb background while retaining the object, and change the object to construct counterfactual training examples. | Foreground/context interventions are established; compare against ordinary context augmentation. |
| Askari Farsangi, Sharifi-Zarchi and Rohban, *Novel Pipeline for Diagnosing Acute Lymphoblastic Leukemia Sensitive to Related Biomarkers*, revised 2025, https://arxiv.org/abs/2307.04014 | Expert-inspired ALL pipeline and multiple-instance learning to address shortcuts. | Cell-aware ALL classification is not a new topic invented here. |
| Huang et al., *Evidence-RL: Towards Evidence-intensive Visual Reasoning*, arXiv:2608.08021, 2026, https://arxiv.org/abs/2608.08021 | Compares erasing evidence with matched nonevidence regions, using that signal in reinforcement learning for vision-language models. | A close and important prior: even area-matched evidence comparison must not be described as a new general principle. It was found after the candidate protocol was written, before fitting completed or new test evaluation. |
| Abdallat et al., *An Artificial Intelligence Framework for Detection of Leukemia with Comparison Between Segmented and Non-Segmented Images*, Sensors 2026, https://www.mdpi.com/1424-8220/26/19/6290 | Compares original and segmented ALL image information. | Merely adding a second view is not the contribution. |

The defensible proposed contribution is an ALL-specific adaptation: automatic stain-region proposals in multicell fields, equal-area region/context erasures, a differentiable class-margin ranking penalty on a frozen CNN representation, and a single-pass classifier. No region-erased example is assigned an invented benign label. Matched augmentation and unmatched ranking controls isolate the contribution of the training term and area control. This is an incremental methodological and empirical contribution, not a first-ever general algorithm. Its usefulness must be established by the recorded comparisons; publication acceptance cannot be promised.

The reused test partition makes the extension exploratory. A new patient-separated or external evaluation would strengthen it. This limitation cannot be fixed by calling a different random split independent.
