import numpy as np


def detect_disorders(cell_counts, feature_list,
                     rbc_features=None, wbc_features=None, blast_features=None):
    """
    Detect hematological disorders based on proposal Section 4.6 rules.
    """

    results = {}
    rbc_features   = rbc_features   or []
    wbc_features   = wbc_features   or []
    blast_features = blast_features or []

    wbc_subtypes  = ['Neutrophil', 'Lymphocyte', 'Monocyte', 'Eosinophil']
    total_wbc     = sum(cell_counts.get(t, 0) for t in wbc_subtypes)
    blast_count   = cell_counts.get('Blast', 0)
    total_rbc     = cell_counts.get('RBC', 0)
    total_wbc_all = total_wbc + blast_count + 1e-10
    total_cells   = sum(cell_counts.values())

    # ── Acute Lymphoblastic Leukemia (ALL) ──────────────────────────
    # Proposal: Lymphoblasts > 20% of WBC
    #           + >10 cells with nuclear irregularity > 1.4
    #           + NC ratio > 0.7 as supporting indicator
    blast_pct = blast_count / total_wbc_all

    all_features_combined = wbc_features + blast_features
    abnormal_nuclei = sum(
        1 for f in all_features_combined
        if f.get('nuclear_irregularity', 0) > 1.4
    )
    high_nc = sum(
        1 for f in blast_features
        if f.get('nc_ratio', 0) > 0.7
    )

    if blast_pct > 0.20 and abnormal_nuclei > 10:
        confidence = min(blast_pct * 1.5 + (abnormal_nuclei / 30), 1.0)
        results['Acute Lymphoblastic Leukemia'] = {
            'detected':   True,
            'confidence': round(confidence, 2),
            'evidence':   (
                f"Blast cells: {blast_count} ({blast_pct*100:.1f}% of WBC), "
                f"{abnormal_nuclei} cells with nuclear irregularity > 1.4, "
                f"{high_nc} cells with NC ratio > 0.7"
            )
        }
    elif blast_count > 0 and (blast_pct > 0.10 or abnormal_nuclei > 5):
        confidence = min(blast_pct + (abnormal_nuclei / 20), 0.75)
        results['Acute Lymphoblastic Leukemia'] = {
            'detected':   True,
            'confidence': round(confidence, 2),
            'evidence':   (
                f"Suspected: {blast_count} blast cells detected "
                f"({blast_pct*100:.1f}% of WBC), "
                f"{abnormal_nuclei} cells with abnormal nuclear irregularity"
            )
        }

    # ── Sickle Cell Disease ─────────────────────────────────────────
    # Proposal: Low circularity of RBCs
    #
    # WHY THE FALSE POSITIVE HAPPENED:
    # Normal RBCs are biconcave discs — when photographed flat under a
    # microscope they naturally appear as irregular ovals/rings, not
    # perfect circles. Their circularity typically ranges 0.60–0.80.
    # The old threshold of 0.75 was too high and caught normal RBCs.
    #
    # FIX:
    # 1. Threshold lowered to 0.60 — true sickle cells are crescent/
    #    elongated and score well below 0.60. Normal RBCs rarely do.
    # 2. Minimum 15 RBCs required — rules out single-cell test images.
    # 3. Percentage raised to 40% — a few deformed RBCs exist even in
    #    healthy blood; real sickle cell disease affects the majority.
    # 4. Added eccentricity check — sickle cells are highly elongated
    #    (eccentricity > 0.8). Normal oval RBCs score ~0.5–0.7.
    #    This is the strongest discriminator.

    if len(rbc_features) >= 15 and total_rbc >= 15 and total_cells >= 20:
        rbc_circularities = [f['circularity'] for f in rbc_features]

        # use 0.60 threshold — true sickle shape is very elongated
        sickle_count = sum(1 for c in rbc_circularities if c < 0.60)
        sickle_pct   = sickle_count / len(rbc_circularities)

        # also check eccentricity if available (highly elongated = sickle)
        eccentric_count = sum(
            1 for f in rbc_features
            if f.get('eccentricity', 0) > 0.80
        )
        eccentric_pct = eccentric_count / len(rbc_features)

        # require BOTH low circularity AND high eccentricity
        # to avoid flagging normal slightly-oval RBCs
        if sickle_pct > 0.40 and eccentric_pct > 0.30:
            confidence = round(min((sickle_pct + eccentric_pct) / 2, 1.0), 2)
            results['Sickle Cell Disease'] = {
                'detected':   True,
                'confidence': confidence,
                'evidence':   (
                    f"{sickle_pct*100:.1f}% of RBCs have circularity < 0.60 "
                    f"({sickle_count}/{len(rbc_circularities)} cells) "
                    f"and {eccentric_pct*100:.1f}% are highly elongated "
                    f"(eccentricity > 0.80), indicating sickle-shaped deformation"
                )
            }

    # ── Anemia ──────────────────────────────────────────────────────
    # Proposal: Abnormalities in RBC size AND color
    # Size: coefficient of variation (CV) > 0.35 = anisocytosis
    # Color: mean NC ratio as proxy for cell paleness (pale RBCs = low NC ratio)
    if len(rbc_features) >= 5 and total_rbc >= 5:
        rbc_areas      = [f['area']        for f in rbc_features]
        rbc_nc_ratios  = [f['nc_ratio']    for f in rbc_features]

        mean_area = np.mean(rbc_areas)
        std_area  = np.std(rbc_areas)
        cv        = std_area / (mean_area + 1e-10)

        # color indicator: very low NC ratio = pale/hypochromic RBC = iron deficiency
        mean_nc   = np.mean(rbc_nc_ratios)
        pale_rbcs = sum(1 for nc in rbc_nc_ratios if nc < 0.15)
        pale_pct  = pale_rbcs / len(rbc_nc_ratios)

        size_abnormal  = cv > 0.35
        color_abnormal = pale_pct > 0.30 or mean_nc < 0.12

        if size_abnormal or color_abnormal:
            evidence_parts = []
            if size_abnormal:
                evidence_parts.append(
                    f"High RBC size variation (CV={cv:.2f}): "
                    f"mean={mean_area:.0f}px, std={std_area:.0f}px"
                )
            if color_abnormal:
                evidence_parts.append(
                    f"Hypochromic RBCs: {pale_pct*100:.1f}% pale cells, "
                    f"mean NC ratio={mean_nc:.2f}"
                )
            results['Anemia'] = {
                'detected':   True,
                'confidence': round(min((cv + pale_pct) / 1.5, 1.0), 2),
                'evidence':   '; '.join(evidence_parts)
            }

    # ── Normal ──────────────────────────────────────────────────────
    if not results:
        results['Normal'] = {
            'detected': False,
            'evidence': 'No hematological abnormalities detected'
        }

    return results
