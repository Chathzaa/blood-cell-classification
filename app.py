# import streamlit as st
# import sys, os, cv2, tempfile
# sys.path.append('src')
# from pipeline import analyze

# st.set_page_config(page_title="Blood Cell Analysis", layout="wide")
# st.title("Automated Blood Cell Classification System")
# st.write("Upload a blood smear image to detect cells and screen for disorders.")

# uploaded = st.file_uploader("Choose a blood smear image",
#                              type=['jpg','jpeg','png','tiff'])

# if uploaded:
#     with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
#         tmp.write(uploaded.read())
#         tmp_path = tmp.name

#     with st.spinner("Analyzing image..."):
#         result, counts, disorders = analyze(tmp_path)

#     col1, col2 = st.columns(2)

#     with col1:
#         st.subheader("Detected cells")
#         for cell_type, count in counts.items():
#             if count > 0:
#                 st.metric(cell_type, count)

#     with col2:
#         st.subheader("Disorder screening")
#         for disorder, info in disorders.items():
#             if disorder == 'Normal':
#                 st.success(f"Normal — {info['evidence']}")
#             else:
#                 st.error(f"FLAG: {disorder}")
#                 st.write(f"Evidence: {info['evidence']}")
#                 st.write(f"Confidence: {info.get('confidence','N/A')}")

#     st.subheader("Annotated image")
#     annotated = result.plot()
#     st.image(annotated, channels="BGR", use_column_width=True)

#     os.unlink(tmp_path)

import streamlit as st
import sys, os, cv2, tempfile, json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime
import pandas as pd

sys.path.append('src')
from pipeline import analyze

# ── page config ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Blood Cell Analysis System",
    page_icon="🔬",
    layout="wide"
)

# ── header ─────────────────────────────────────────────────────────────
st.markdown("# 🔬 Automated Blood Cell Classification System")
st.markdown("**University of Ruhuna — EE7204 / EC7205 Image Processing & Computer Vision**")
st.divider()

# ── sidebar ────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("About")
    st.info(
        "This system automatically analyzes blood smear images to:\n\n"
        "• Detect and classify blood cells\n"
        "• Count RBCs, WBCs, Platelets\n"
        "• Screen for hematological disorders\n"
        "• Generate clinical reports"
    )
    st.header("Normal Ranges")
    st.markdown("""
    | Cell Type | Normal % |
    |-----------|----------|
    | Neutrophil | 50–70% |
    | Lymphocyte | 20–40% |
    | Monocyte | 2–8% |
    | Eosinophil | 1–4% |
    """)

# ── upload ─────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload a blood smear microscope image",
    type=['jpg', 'jpeg', 'png', 'tiff'],
    help="Supports JPEG, PNG, TIFF formats from microscope"
)

if uploaded:
    # save temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    with st.spinner("🔬 Analyzing blood smear..."):
        result, counts, disorders, feature_list = analyze(tmp_path)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── TAB LAYOUT ─────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Cell Analysis",
        "🖼️ Visual Results",
        "⚠️ Disorder Screening",
        "📄 Clinical Report"
    ])

    # ══════════════════════════════════════════════════════════════════
    # TAB 1 — CELL ANALYSIS
    # ══════════════════════════════════════════════════════════════════
    with tab1:
        st.subheader("Complete Blood Count (CBC)")

        # total counts row
        total_rbc      = counts.get('RBC', 0)
        total_wbc      = sum(counts.get(t, 0) for t in
                             ['Neutrophil','Lymphocyte','Monocyte','Eosinophil','WBC'])
        total_platelet = counts.get('Platelet', 0)
        total_cells    = sum(counts.values())

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🔴 Red Blood Cells",  total_rbc,      help="RBC count")
        c2.metric("⚪ White Blood Cells", total_wbc,      help="Total WBC count")
        c3.metric("🟡 Platelets",         total_platelet, help="Platelet count")
        c4.metric("🔢 Total Cells",        total_cells,    help="All detected cells")

        st.divider()

        # WBC differential
        st.subheader("WBC Differential Count")
        wbc_types = ['Neutrophil', 'Lymphocyte', 'Monocyte', 'Eosinophil']
        wbc_counts = {t: counts.get(t, 0) for t in wbc_types}
        total_wbc_diff = sum(wbc_counts.values()) + 1e-10

        diff_data = []
        normal_ranges = {
            'Neutrophil':  (50, 70),
            'Lymphocyte':  (20, 40),
            'Monocyte':    (2,  8),
            'Eosinophil':  (1,  4),
        }
        for cell, cnt in wbc_counts.items():
            pct = round((cnt / total_wbc_diff) * 100, 1)
            low, high = normal_ranges[cell]
            if pct < low:
                status = "🔵 Low"
            elif pct > high:
                status = "🔴 High"
            else:
                status = "🟢 Normal"
            diff_data.append({
                'Cell Type':   cell,
                'Count':       cnt,
                'Percentage':  f"{pct}%",
                'Normal Range': f"{low}–{high}%",
                'Status':      status
            })

        df_diff = pd.DataFrame(diff_data)
        st.dataframe(df_diff, use_container_width=True, hide_index=True)

        # pie chart of WBC differential
        if total_wbc > 0:
            fig, ax = plt.subplots(figsize=(5, 4))
            sizes  = [wbc_counts[t] for t in wbc_types if wbc_counts[t] > 0]
            labels = [t for t in wbc_types if wbc_counts[t] > 0]
            colors = ['#4C72B0','#55A868','#C44E52','#8172B2'][:len(labels)]
            ax.pie(sizes, labels=labels, colors=colors,
                   autopct='%1.1f%%', startangle=90)
            ax.set_title('WBC Differential')
            st.pyplot(fig)
            plt.close()

    # ══════════════════════════════════════════════════════════════════
    # TAB 2 — VISUAL RESULTS
    # ══════════════════════════════════════════════════════════════════
    with tab2:
        st.subheader("Detection Results")

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("**Original Image**")
            original = cv2.imread(tmp_path)
            st.image(cv2.cvtColor(original, cv2.COLOR_BGR2RGB),
                     use_column_width=True)

        with col_b:
            st.markdown("**Annotated — Detected Cells with Bounding Boxes**")
            annotated = result.plot()
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                     use_column_width=True)

        # save annotated image
        ann_path = tmp_path.replace('.jpg', '_annotated.jpg')
        cv2.imwrite(ann_path, annotated)

        st.divider()
        st.subheader("Morphological Feature Distribution")

        if feature_list:
            areas        = [f['area']        for f in feature_list]
            circularities= [f['circularity'] for f in feature_list]

            fig2, axes = plt.subplots(1, 2, figsize=(12, 4))

            axes[0].hist(areas, bins=20, color='#4C72B0', edgecolor='white')
            axes[0].set_title('Cell Area Distribution')
            axes[0].set_xlabel('Area (pixels)')
            axes[0].set_ylabel('Count')
            axes[0].axvline(np.mean(areas), color='red',
                            linestyle='--', label=f'Mean: {np.mean(areas):.0f}')
            axes[0].legend()

            axes[1].hist(circularities, bins=20,
                         color='#55A868', edgecolor='white')
            axes[1].set_title('Cell Circularity Distribution')
            axes[1].set_xlabel('Circularity (0–1)')
            axes[1].set_ylabel('Count')
            axes[1].axvline(np.mean(circularities), color='red',
                            linestyle='--',
                            label=f'Mean: {np.mean(circularities):.2f}')
            axes[1].legend()

            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()

            # feature stats table
            st.subheader("Morphological Statistics Summary")
            stats = {
                'Feature': ['Area', 'Circularity',
                             'Nuclear Irregularity', 'NC Ratio'],
                'Mean':    [
                    round(np.mean([f['area'] for f in feature_list]), 2),
                    round(np.mean([f['circularity'] for f in feature_list]), 3),
                    round(np.mean([f['nuclear_irregularity'] for f in feature_list]), 3),
                    round(np.mean([f['nc_ratio'] for f in feature_list]), 3),
                ],
                'Std Dev': [
                    round(np.std([f['area'] for f in feature_list]), 2),
                    round(np.std([f['circularity'] for f in feature_list]), 3),
                    round(np.std([f['nuclear_irregularity'] for f in feature_list]), 3),
                    round(np.std([f['nc_ratio'] for f in feature_list]), 3),
                ],
                'Min': [
                    round(min(f['area'] for f in feature_list), 2),
                    round(min(f['circularity'] for f in feature_list), 3),
                    round(min(f['nuclear_irregularity'] for f in feature_list), 3),
                    round(min(f['nc_ratio'] for f in feature_list), 3),
                ],
                'Max': [
                    round(max(f['area'] for f in feature_list), 2),
                    round(max(f['circularity'] for f in feature_list), 3),
                    round(max(f['nuclear_irregularity'] for f in feature_list), 3),
                    round(max(f['nc_ratio'] for f in feature_list), 3),
                ],
            }
            st.dataframe(pd.DataFrame(stats),
                         use_container_width=True, hide_index=True)

    # ══════════════════════════════════════════════════════════════════
    # TAB 3 — DISORDER SCREENING
    # ══════════════════════════════════════════════════════════════════
    with tab3:
        st.subheader("Hematological Disorder Screening")
        st.caption("Rule-based analysis using morphological features and cell population statistics")

        for disorder, info in disorders.items():
            if disorder == 'Normal':
                st.success("✅ No hematological disorders detected")
                st.info(f"Evidence: {info['evidence']}")
            else:
                with st.expander(f"⚠️ FLAG: {disorder}", expanded=True):
                    col_x, col_y = st.columns(2)
                    with col_x:
                        st.error(f"**Disorder:** {disorder}")
                        st.write(f"**Evidence:** {info['evidence']}")
                        if 'confidence' in info:
                            conf = float(info['confidence'])
                            st.write(f"**Confidence:** {conf*100:.1f}%")
                            st.progress(conf)
                    with col_y:
                        # detection criteria from proposal
                        criteria = {
                            'Acute Lymphoblastic Leukemia':
                                "Lymphoblasts > 20% of WBC\nNuclear irregularity > 1.4\nHigh NC ratio > 0.7",
                            'Sickle Cell Disease':
                                "RBC circularity < 0.85\n> 5% of RBCs affected",
                            'Anemia':
                                "RBC size deviation > 2σ from normal\nAbnormal size distribution",
                        }
                        crit = criteria.get(disorder, "See clinical guidelines")
                        st.markdown("**Detection Criteria Used:**")
                        st.code(crit)

        st.divider()
        st.subheader("Screening Summary")
        screen_data = []
        for disorder, info in disorders.items():
            screen_data.append({
                'Disorder':  disorder,
                'Detected':  '✅ Yes' if info.get('detected') else '❌ No',
                'Evidence':  info['evidence'],
                'Confidence': f"{float(info.get('confidence',0))*100:.1f}%"
                              if 'confidence' in info else 'N/A'
            })
        st.dataframe(pd.DataFrame(screen_data),
                     use_container_width=True, hide_index=True)

    # ══════════════════════════════════════════════════════════════════
    # TAB 4 — CLINICAL REPORT
    # ══════════════════════════════════════════════════════════════════
    with tab4:
        st.subheader("Clinical Report")
        st.caption(f"Generated: {timestamp}")

        # build report dict
        report = {
            "report_metadata": {
                "generated_at":  timestamp,
                "system":        "Automated Blood Cell Classification System",
                "institution":   "University of Ruhuna",
                "module":        "EE7204 / EC7205"
            },
            "cell_counts": {
                "RBC":       total_rbc,
                "WBC_total": total_wbc,
                "Platelet":  total_platelet,
                "differential": wbc_counts
            },
            "wbc_differential_percentages": {
                t: round((wbc_counts[t] / total_wbc_diff) * 100, 1)
                for t in wbc_types
            },
            "morphological_summary": {
                "total_cells_analyzed": len(feature_list),
                "mean_area":      round(np.mean([f['area'] for f in feature_list]), 2) if feature_list else 0,
                "mean_circularity": round(np.mean([f['circularity'] for f in feature_list]), 3) if feature_list else 0,
                "mean_nc_ratio":  round(np.mean([f['nc_ratio'] for f in feature_list]), 3) if feature_list else 0,
            },
            "disorder_screening": disorders,
            "abnormal_flags": [d for d, i in disorders.items()
                               if d != 'Normal' and i.get('detected')]
        }

        # display as formatted report
        st.markdown("### Patient Report")
        st.markdown(f"**Date/Time:** {timestamp}")
        st.markdown(f"**Image File:** {uploaded.name}")
        st.divider()

        st.markdown("#### CBC Results")
        rpt_cols = st.columns(3)
        rpt_cols[0].metric("RBC Count",     total_rbc)
        rpt_cols[1].metric("WBC Count",     total_wbc)
        rpt_cols[2].metric("Platelet Count", total_platelet)

        st.markdown("#### WBC Differential")
        st.dataframe(df_diff, use_container_width=True, hide_index=True)

        st.markdown("#### Morphological Summary")
        morph_df = pd.DataFrame([{
            'Total Cells Analyzed': len(feature_list),
            'Mean Cell Area (px)':  report['morphological_summary']['mean_area'],
            'Mean Circularity':     report['morphological_summary']['mean_circularity'],
            'Mean NC Ratio':        report['morphological_summary']['mean_nc_ratio'],
        }])
        st.dataframe(morph_df, use_container_width=True, hide_index=True)

        st.markdown("#### Disorder Flags")
        if report['abnormal_flags']:
            for flag in report['abnormal_flags']:
                st.error(f"⚠️ {flag}")
        else:
            st.success("✅ No abnormalities flagged")

        st.divider()

        # ── DOWNLOAD BUTTONS ───────────────────────────────────────
        st.subheader("Download Report")
        dl1, dl2, dl3 = st.columns(3)

        # JSON download
        json_str = json.dumps(report, indent=2)
        dl1.download_button(
            label="📥 Download JSON",
            data=json_str,
            file_name=f"blood_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

        # CSV download
        csv_rows = []
        for cell, cnt in counts.items():
            csv_rows.append({'Category': 'Cell Count',
                             'Item': cell, 'Value': cnt, 'Unit': 'cells'})
        for t in wbc_types:
            pct = round((wbc_counts[t] / total_wbc_diff) * 100, 1)
            csv_rows.append({'Category': 'WBC Differential',
                             'Item': t, 'Value': pct, 'Unit': '%'})
        for d, info in disorders.items():
            csv_rows.append({'Category': 'Disorder Screening',
                             'Item': d,
                             'Value': info['evidence'],
                             'Unit': ''})
        csv_str = pd.DataFrame(csv_rows).to_csv(index=False)
        dl2.download_button(
            label="📥 Download CSV",
            data=csv_str,
            file_name=f"blood_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

        # annotated image download
        with open(ann_path, 'rb') as f:
            dl3.download_button(
                label="📥 Download Annotated Image",
                data=f,
                file_name=f"annotated_{uploaded.name}",
                mime="image/jpeg"
            )

    os.unlink(tmp_path)