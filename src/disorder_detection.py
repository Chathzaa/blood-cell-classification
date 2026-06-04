import numpy as np

def detect_disorders(cell_counts, feature_list):
    results = {}
    wbc_types = ['neutrophil','lymphocyte','monocyte','eosinophil']
    total_wbc = sum(cell_counts.get(t, 0) for t in wbc_types) + 1e-10

    # --- Acute Lymphoblastic Leukemia ---
    lymphoblast_pct = cell_counts.get('lymphoblast', 0) / total_wbc
    blast_count = sum(1 for f in feature_list
                      if f.get('nuclear_irregularity', 0) > 1.4)
    if lymphoblast_pct > 0.20 and blast_count > 10:
        results['Acute Lymphoblastic Leukemia'] = {
            'detected': True,
            'confidence': round(min(lymphoblast_pct * 2, 1.0), 2),
            'evidence': f"{lymphoblast_pct*100:.1f}% lymphoblasts, "
                        f"{blast_count} abnormal nuclei"
        }

    # --- Sickle Cell Disease ---
    rbc_circularities = [f['circularity'] for f in feature_list
                         if f.get('cell_type') == 'RBC']
    if rbc_circularities:
        sickle_pct = sum(1 for c in rbc_circularities if c < 0.85) \
                     / len(rbc_circularities)
        if sickle_pct > 0.05:
            results['Sickle Cell Disease'] = {
                'detected': True,
                'evidence': f"{sickle_pct*100:.1f}% deformed RBCs"
            }

    # --- Anemia ---
    rbc_areas = [f['area'] for f in feature_list
                 if f.get('cell_type') == 'RBC']
    if rbc_areas:
        mean_area = np.mean(rbc_areas)
        std_area  = np.std(rbc_areas)
        normal_rbc = 250
        if abs(mean_area - normal_rbc) > 2 * std_area:
            results['Anemia'] = {
                'detected': True,
                'evidence': f"Abnormal RBC size: mean={mean_area:.0f}px"
            }

    if not results:
        results['Normal'] = {'detected': False,
                             'evidence': 'No abnormalities detected'}
    return results