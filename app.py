import streamlit as st
import sys, os, cv2, tempfile, json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
from io import BytesIO

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


def generate_pdf(report, annotated_img, timestamp, filename):
    """Generate PDF clinical report using reportlab."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                         Table, TableStyle, Image as RLImage,
                                         HRFlowable)
        from reportlab.lib.enums import TA_CENTER, TA_LEFT

        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4,
                                 rightMargin=2*cm, leftMargin=2*cm,
                                 topMargin=2*cm, bottomMargin=2*cm)

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle('Title', parent=styles['Title'],
                                      fontSize=16, spaceAfter=6,
                                      textColor=colors.HexColor('#1a237e'))
        heading_style = ParagraphStyle('Heading', parent=styles['Heading2'],
                                        fontSize=12, spaceAfter=4,
                                        textColor=colors.HexColor('#1565c0'))
        normal_style = styles['Normal']
        center_style = ParagraphStyle('Center', parent=styles['Normal'],
                                       alignment=TA_CENTER)

        story = []

        # Title
        story.append(Paragraph("Automated Blood Cell Analysis Report", title_style))
        story.append(Paragraph("University of Ruhuna — EE7204 / EC7205", center_style))
        story.append(Spacer(1, 0.3*cm))
        story.append(HRFlowable(width="100%", thickness=2,
                                  color=colors.HexColor('#1a237e')))
        story.append(Spacer(1, 0.3*cm))

        # Metadata
        story.append(Paragraph("Report Information", heading_style))
        meta_data = [
            ['Generated At', timestamp],
            ['Image File',   filename],
            ['Institution',  'University of Ruhuna'],
            ['Module',       'EE7204 / EC7205 Image Processing & Computer Vision'],
        ]
        meta_table = Table(meta_data, colWidths=[5*cm, 12*cm])
        meta_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#e3f2fd')),
            ('FONTNAME',   (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE',   (0, 0), (-1, -1), 9),
            ('GRID',       (0, 0), (-1, -1), 0.5, colors.grey),
            ('PADDING',    (0, 0), (-1, -1), 4),
        ]))
        story.append(meta_table)
        story.append(Spacer(1, 0.4*cm))

        # Annotated image
        story.append(Paragraph("Annotated Blood Smear", heading_style))
        img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        img_buf = BytesIO()
        from PIL import Image as PILImage
        PILImage.fromarray(img_rgb).save(img_buf, format='JPEG', quality=85)
        img_buf.seek(0)
        rl_img = RLImage(img_buf, width=14*cm, height=10*cm)
        story.append(rl_img)
        story.append(Spacer(1, 0.4*cm))

        # CBC Results
        story.append(Paragraph("Complete Blood Count (CBC)", heading_style))
        cc = report['cell_counts']
        cbc_data = [
            ['Cell Type', 'Count'],
            ['Red Blood Cells (RBC)',   str(cc['RBC'])],
            ['White Blood Cells (WBC)', str(cc['WBC_total'])],
            ['Platelets',               str(cc['Platelet'])],
        ]
        cbc_table = Table(cbc_data, colWidths=[10*cm, 7*cm])
        cbc_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1565c0')),
            ('TEXTCOLOR',  (0, 0), (-1, 0), colors.white),
            ('FONTNAME',   (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE',   (0, 0), (-1, -1), 10),
            ('GRID',       (0, 0), (-1, -1), 0.5, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1),
             [colors.white, colors.HexColor('#f5f5f5')]),
            ('PADDING',    (0, 0), (-1, -1), 6),
        ]))
        story.append(cbc_table)
        story.append(Spacer(1, 0.4*cm))

        # WBC Differential
        story.append(Paragraph("WBC Differential Count", heading_style))
        diff_pct = report['wbc_differential_percentages']
        normal_ranges = {
            'Neutrophil': (50, 70), 'Lymphocyte': (20, 40),
            'Monocyte': (2, 8),     'Eosinophil': (1, 4),
        }
        diff_data = [['Cell Type', 'Count', 'Percentage', 'Normal Range', 'Status']]
        for cell in ['Neutrophil', 'Lymphocyte', 'Monocyte', 'Eosinophil']:
            cnt = cc['differential'].get(cell, 0)
            pct = diff_pct.get(cell, 0.0)
            low, high = normal_ranges[cell]
            if pct < low:   status = 'Low'
            elif pct > high: status = 'High'
            else:            status = 'Normal'
            diff_data.append([cell, str(cnt), f"{pct}%",
                               f"{low}–{high}%", status])

        diff_table = Table(diff_data, colWidths=[4*cm, 3*cm, 3*cm, 4*cm, 3*cm])
        diff_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1565c0')),
            ('TEXTCOLOR',  (0, 0), (-1, 0), colors.white),
            ('FONTNAME',   (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE',   (0, 0), (-1, -1), 9),
            ('GRID',       (0, 0), (-1, -1), 0.5, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1),
             [colors.white, colors.HexColor('#f5f5f5')]),
            ('PADDING',    (0, 0), (-1, -1), 5),
        ]))
        story.append(diff_table)
        story.append(Spacer(1, 0.4*cm))

        # Morphological Summary
        story.append(Paragraph("Morphological Summary", heading_style))
        morph = report['morphological_summary']
        morph_data = [
            ['Parameter', 'Value'],
            ['Total Cells Analyzed', str(morph['total_cells_analyzed'])],
            ['Mean Cell Area (px)',  str(morph['mean_area'])],
            ['Mean Circularity',     str(morph['mean_circularity'])],
            ['Mean NC Ratio',        str(morph['mean_nc_ratio'])],
        ]
        morph_table = Table(morph_data, colWidths=[9*cm, 8*cm])
        morph_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1565c0')),
            ('TEXTCOLOR',  (0, 0), (-1, 0), colors.white),
            ('FONTNAME',   (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE',   (0, 0), (-1, -1), 9),
            ('GRID',       (0, 0), (-1, -1), 0.5, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1),
             [colors.white, colors.HexColor('#f5f5f5')]),
            ('PADDING',    (0, 0), (-1, -1), 5),
        ]))
        story.append(morph_table)
        story.append(Spacer(1, 0.4*cm))

        # Disorder Screening
        story.append(Paragraph("Disorder Screening Results", heading_style))
        ds = report['disorder_screening']
        flags = report['abnormal_flags']

        if not flags:
            story.append(Paragraph(
                "✓ No hematological abnormalities detected", normal_style))
        else:
            for disorder in flags:
                info = ds.get(disorder, {})
                conf = info.get('confidence', 0)
                story.append(Paragraph(
                    f"⚠ FLAGGED: {disorder}", heading_style))
                disorder_data = [
                    ['Evidence',   info.get('evidence', 'N/A')],
                    ['Confidence', f"{float(conf)*100:.1f}%"],
                ]
                d_table = Table(disorder_data, colWidths=[4*cm, 13*cm])
                d_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#ffebee')),
                    ('FONTNAME',   (0, 0), (0, -1), 'Helvetica-Bold'),
                    ('FONTSIZE',   (0, 0), (-1, -1), 9),
                    ('GRID',       (0, 0), (-1, -1), 0.5, colors.grey),
                    ('PADDING',    (0, 0), (-1, -1), 5),
                ]))
                story.append(d_table)
                story.append(Spacer(1, 0.2*cm))

        # Footer
        story.append(Spacer(1, 0.5*cm))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
        story.append(Spacer(1, 0.2*cm))
        story.append(Paragraph(
            "This report is generated by an automated system for research purposes only. "
            "Clinical decisions should be made by qualified medical professionals.",
            ParagraphStyle('Disclaimer', parent=styles['Normal'],
                           fontSize=8, textColor=colors.grey)
        ))

        doc.build(story)
        buffer.seek(0)
        return buffer

    except ImportError:
        return None


# ── upload ─────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload a blood smear microscope image",
    type=['jpg', 'jpeg', 'png', 'tiff'],
    help="Supports JPEG, PNG, TIFF formats from microscope"
)

if uploaded:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    with st.spinner("🔬 Analyzing blood smear..."):
        result, counts, disorders, feature_list = analyze(tmp_path)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

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

        total_rbc      = counts.get('RBC', 0)
        total_wbc      = sum(counts.get(t, 0) for t in
                             ['Neutrophil','Lymphocyte','Monocyte','Eosinophil','WBC'])
        total_platelet = counts.get('Platelet', 0)
        total_cells    = sum(counts.values())

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🔴 Red Blood Cells",   total_rbc,      help="RBC count")
        c2.metric("⚪ White Blood Cells",  total_wbc,      help="Total WBC count")
        c3.metric("🟡 Platelets",          total_platelet, help="Platelet count")
        c4.metric("🔢 Total Cells",         total_cells,    help="All detected cells")

        st.divider()

        st.subheader("WBC Differential Count")
        wbc_types  = ['Neutrophil', 'Lymphocyte', 'Monocyte', 'Eosinophil']
        wbc_counts = {t: counts.get(t, 0) for t in wbc_types}
        total_wbc_diff = sum(wbc_counts.values()) + 1e-10

        normal_ranges = {
            'Neutrophil': (50, 70), 'Lymphocyte': (20, 40),
            'Monocyte':   (2,  8),  'Eosinophil': (1,  4),
        }
        diff_data = []
        for cell, cnt in wbc_counts.items():
            pct = round((cnt / total_wbc_diff) * 100, 1)
            low, high = normal_ranges[cell]
            status = "🔵 Low" if pct < low else ("🔴 High" if pct > high else "🟢 Normal")
            diff_data.append({
                'Cell Type':    cell,
                'Count':        cnt,
                'Percentage':   f"{pct}%",
                'Normal Range': f"{low}–{high}%",
                'Status':       status
            })

        df_diff = pd.DataFrame(diff_data)
        st.dataframe(df_diff, use_container_width=True, hide_index=True)

        if total_wbc > 0:
            fig, ax = plt.subplots(figsize=(5, 4))
            sizes  = [wbc_counts[t] for t in wbc_types if wbc_counts[t] > 0]
            labels = [t for t in wbc_types if wbc_counts[t] > 0]
            colors_pie = ['#4C72B0','#55A868','#C44E52','#8172B2'][:len(labels)]
            ax.pie(sizes, labels=labels, colors=colors_pie,
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
                     use_container_width=True)
        with col_b:
            st.markdown("**Annotated — Detected Cells with Bounding Boxes**")
            annotated = result.plot()
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                     use_container_width=True)

        ann_path = tmp_path.replace('.jpg', '_annotated.jpg')
        cv2.imwrite(ann_path, annotated)

        st.divider()
        st.subheader("Morphological Feature Distribution")

        if feature_list:
            areas         = [f['area']        for f in feature_list]
            circularities = [f['circularity'] for f in feature_list]

            fig2, axes = plt.subplots(1, 2, figsize=(12, 4))
            axes[0].hist(areas, bins=20, color='#4C72B0', edgecolor='white')
            axes[0].set_title('Cell Area Distribution')
            axes[0].set_xlabel('Area (pixels)')
            axes[0].set_ylabel('Count')
            axes[0].axvline(np.mean(areas), color='red', linestyle='--',
                            label=f'Mean: {np.mean(areas):.0f}')
            axes[0].legend()

            axes[1].hist(circularities, bins=20, color='#55A868', edgecolor='white')
            axes[1].set_title('Cell Circularity Distribution')
            axes[1].set_xlabel('Circularity (0–1)')
            axes[1].set_ylabel('Count')
            axes[1].axvline(np.mean(circularities), color='red', linestyle='--',
                            label=f'Mean: {np.mean(circularities):.2f}')
            axes[1].legend()

            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()

            st.subheader("Morphological Statistics Summary")
            stats = {
                'Feature':  ['Area', 'Circularity', 'Nuclear Irregularity', 'NC Ratio'],
                'Mean':     [round(np.mean([f['area']                 for f in feature_list]), 2),
                             round(np.mean([f['circularity']          for f in feature_list]), 3),
                             round(np.mean([f['nuclear_irregularity'] for f in feature_list]), 3),
                             round(np.mean([f['nc_ratio']             for f in feature_list]), 3)],
                'Std Dev':  [round(np.std([f['area']                  for f in feature_list]), 2),
                             round(np.std([f['circularity']           for f in feature_list]), 3),
                             round(np.std([f['nuclear_irregularity']  for f in feature_list]), 3),
                             round(np.std([f['nc_ratio']              for f in feature_list]), 3)],
                'Min':      [round(min(f['area']                      for f in feature_list), 2),
                             round(min(f['circularity']               for f in feature_list), 3),
                             round(min(f['nuclear_irregularity']      for f in feature_list), 3),
                             round(min(f['nc_ratio']                  for f in feature_list), 3)],
                'Max':      [round(max(f['area']                      for f in feature_list), 2),
                             round(max(f['circularity']               for f in feature_list), 3),
                             round(max(f['nuclear_irregularity']      for f in feature_list), 3),
                             round(max(f['nc_ratio']                  for f in feature_list), 3)],
            }
            st.dataframe(pd.DataFrame(stats), use_container_width=True, hide_index=True)

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
                        criteria = {
                            'Acute Lymphoblastic Leukemia':
                                "Lymphoblasts > 20% of WBC\nNuclear irregularity > 1.4\nHigh NC ratio > 0.7",
                            'Sickle Cell Disease':
                                "RBC circularity < 0.75\n> 20% of RBCs affected",
                            'Anemia':
                                "High RBC size variation (CV > 0.35)\nOR hypochromic RBCs (pale cells > 30%)",
                        }
                        crit = criteria.get(disorder, "See clinical guidelines")
                        st.markdown("**Detection Criteria Used:**")
                        st.code(crit)

        st.divider()
        st.subheader("Screening Summary")
        screen_data = []
        for disorder, info in disorders.items():
            screen_data.append({
                'Disorder':   disorder,
                'Detected':   '✅ Yes' if info.get('detected') else '❌ No',
                'Evidence':   info['evidence'],
                'Confidence': f"{float(info.get('confidence', 0))*100:.1f}%"
                              if 'confidence' in info else 'N/A'
            })
        st.dataframe(pd.DataFrame(screen_data), use_container_width=True, hide_index=True)

    # ══════════════════════════════════════════════════════════════════
    # TAB 4 — CLINICAL REPORT
    # ══════════════════════════════════════════════════════════════════
    with tab4:
        st.subheader("Clinical Report")
        st.caption(f"Generated: {timestamp}")

        report = {
            "report_metadata": {
                "generated_at": timestamp,
                "system":       "Automated Blood Cell Classification System",
                "institution":  "University of Ruhuna",
                "module":       "EE7204 / EC7205"
            },
            "cell_counts": {
                "RBC":          total_rbc,
                "WBC_total":    total_wbc,
                "Platelet":     total_platelet,
                "differential": wbc_counts
            },
            "wbc_differential_percentages": {
                t: round((wbc_counts[t] / total_wbc_diff) * 100, 1)
                for t in wbc_types
            },
            "morphological_summary": {
                "total_cells_analyzed": len(feature_list),
                "mean_area":       round(np.mean([f['area']        for f in feature_list]), 2) if feature_list else 0,
                "mean_circularity":round(np.mean([f['circularity'] for f in feature_list]), 3) if feature_list else 0,
                "mean_nc_ratio":   round(np.mean([f['nc_ratio']    for f in feature_list]), 3) if feature_list else 0,
            },
            "disorder_screening": disorders,
            "abnormal_flags": [d for d, i in disorders.items()
                               if d != 'Normal' and i.get('detected')]
        }

        st.markdown("### Patient Report")
        st.markdown(f"**Date/Time:** {timestamp}")
        st.markdown(f"**Image File:** {uploaded.name}")
        st.divider()

        st.markdown("#### CBC Results")
        rpt_cols = st.columns(3)
        rpt_cols[0].metric("RBC Count",      total_rbc)
        rpt_cols[1].metric("WBC Count",      total_wbc)
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
        dl1, dl2, dl3, dl4 = st.columns(4)

        # JSON
        json_str = json.dumps(report, indent=2)
        dl1.download_button(
            label="📥 JSON",
            data=json_str,
            file_name=f"blood_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

        # CSV
        csv_rows = []
        for cell, cnt in counts.items():
            csv_rows.append({'Category': 'Cell Count', 'Item': cell,
                             'Value': cnt, 'Unit': 'cells'})
        for t in wbc_types:
            pct = round((wbc_counts[t] / total_wbc_diff) * 100, 1)
            csv_rows.append({'Category': 'WBC Differential', 'Item': t,
                             'Value': pct, 'Unit': '%'})
        for d, info in disorders.items():
            csv_rows.append({'Category': 'Disorder Screening', 'Item': d,
                             'Value': info['evidence'], 'Unit': ''})
        csv_str = pd.DataFrame(csv_rows).to_csv(index=False)
        dl2.download_button(
            label="📥 CSV",
            data=csv_str,
            file_name=f"blood_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

        # PDF
        pdf_buf = generate_pdf(report, annotated, timestamp, uploaded.name)
        if pdf_buf:
            dl3.download_button(
                label="📥 PDF Report",
                data=pdf_buf,
                file_name=f"blood_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf"
            )
        else:
            dl3.warning("Install reportlab for PDF: pip install reportlab pillow")

        # Annotated image
        with open(ann_path, 'rb') as f:
            dl4.download_button(
                label="📥 Annotated Image",
                data=f,
                file_name=f"annotated_{uploaded.name}",
                mime="image/jpeg"
            )

    os.unlink(tmp_path)
