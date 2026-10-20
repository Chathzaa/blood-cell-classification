"""Blood cell detection and ALL screening research workbench."""
from pathlib import Path
import json
import os
import tempfile
import cv2
import numpy as np
import pandas as pd
import streamlit as st

ROOT=Path(__file__).resolve().parent
os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'audit'))
from src.pipeline import analyze,annotate,classify_single
from src.leukemia_screening import screen_all,analyze_combined
from src.all_views import prepare_view
from src.evidence_regularization import intervention_views
from src.evidence_screening import screen_all as screen_evidence,analyze_combined as analyze_evidence_combined

st.set_page_config(page_title='Blood Cells and Leukemia Screening',layout='wide')
st.title('Blood Cells and Leukemia Screening')
st.caption('Cell detection, acute lymphoblastic leukemia screening and review of uncertain results.')
st.info('Research prototype. Leukemia screening is limited to ALL image patterns and does not establish a clinical diagnosis. Image counts are not a complete blood count.')

with st.sidebar:
    mode=st.radio('Analysis',['Blood cells + ALL screening','Blood cell analysis','Single WBC crop','Leukemia screening (ALL)'])
    experimental=st.checkbox('Use experimental evidence training',value=False,
        help='Research comparison only. The AMER candidate did not improve clean-image accuracy and performed worse under global colour changes.') if mode in ('Blood cells + ALL screening','Leukemia screening (ALL)') else False
    if mode=='Blood cells + ALL screening':
        st.caption('Shows cell detections and a separate whole-image ALL assessment. The two components were evaluated on different datasets; combined accuracy has not been established.')
    elif mode=='Leukemia screening (ALL)':
        st.caption('Screens for ALL image patterns and sends uncertain results or subtypes for review.')
    if experimental:
        st.caption('Experimental AMER: trained to compare stained regions with equally large context regions. Its test results do not support replacing the default model.')
    st.markdown('**Study results**')
    for name,file in [('ALL screening study','outputs/all_verified/study.json'),('Evidence-training experiment','outputs/all_novel/study.json'),('Blood cell study','outputs/verified/summary.json')]:
        p=ROOT/file
        if p.exists():st.download_button(name,p.read_bytes(),p.name,'application/json')


def show_cells(image,report,show_subtypes):
    counts=report['cell_counts']
    for col,key,label in zip(st.columns(4),['RBC','WBC','Platelet','total'],
                             ['Red blood cells','White blood cells','Platelets','Total detected']):
        col.metric(label,counts[key])
    if show_subtypes:
        preview=annotate(image,report)
    else:
        preview=image.copy()
        for r in report['detections']:
            x1,y1,x2,y2=r['box']
            color={'RBC':(70,130,220),'WBC':(220,130,30),'Platelet':(80,160,70)}[r['cell_type']]
            cv2.rectangle(preview,(x1,y1),(x2,y2),color,2)
            cv2.putText(preview,r['cell_type'],(x1,max(12,y1-5)),cv2.FONT_HERSHEY_SIMPLEX,.4,color,1,cv2.LINE_AA)
    st.image(cv2.cvtColor(preview,cv2.COLOR_BGR2RGB),caption='Detected objects in this image.',width='stretch')
    if show_subtypes:
        st.dataframe(pd.DataFrame([{'Subtype':c.title(),'Accepted count':n} for c,n in counts['wbc_subtypes'].items()]),hide_index=True)
        st.write(f"White blood cells awaiting subtype review: {counts['wbc_review']}")
    else:
        st.caption('Normal WBC subtype labels are withheld in combined leukemia mode.')
    with st.expander('Descriptive cell measurements'):
        rows=[]
        for r in report['detections']:
            m=r['morphology']
            if m['available']:
                f=m['features']
                rows.append({'Object':r['object_id'],'Type':r['cell_type'],'Area (square pixels)':f['area'],
                             'Circularity':f['circularity'],'High saturation fraction':f['high_saturation_fraction']})
        st.caption('Measurements describe thresholded image regions; they are not validated cell or nucleus boundaries.')
        if rows:st.dataframe(pd.DataFrame(rows),hide_index=True)
    ok,encoded=cv2.imencode('.png',preview)
    if ok:st.download_button('Download cell annotation',encoded.tobytes(),'detected_cells.png','image/png')


def show_all(image,result):
    st.subheader('Leukemia screening')
    if result['status']=='review':
        st.warning(result['label'])
    elif result['status']=='all_pattern':
        st.warning(result['label'])
    else:
        st.success(result['label'])
    st.write(f"Model support for ALL: {result['all_model_support']:.3f}")
    if result['subtype_accepted']:
        st.write('Supported dataset subtype: '+result['subtype_display'])
    elif result['subtype_status']=='review':
        st.write('The ALL screening flag is retained; the subtype requires review.')
    st.caption('Benign means a benign-like dataset pattern. Early pre-B, Pre-B and Pro-B are subtype categories, not severity levels. Other leukemia types are not assessed.')
    if result.get('method')=='area_matched_evidence_training':
        with st.expander('Inspect the experimental training idea'):
            views,masks=intervention_views(image)
            for col,key,label in zip(st.columns(3),['original','context','region'],['Original','Context erased','Stained region erased']):
                col.image(cv2.cvtColor(views[key],cv2.COLOR_BGR2RGB),caption=label,width='stretch')
            st.dataframe(pd.DataFrame(result['views']).rename_axis('Dataset category'),width='stretch')
            st.caption('Screening uses the original image only. These automatic region proposals illustrate the training rule; they are not labelled malignant cells.')
    else:
        with st.expander('Compare the two image views'):
            a,b=st.columns(2)
            a.image(cv2.cvtColor(prepare_view(image),cv2.COLOR_BGR2RGB),caption='Original view',width='stretch')
            b.image(cv2.cvtColor(prepare_view(image,'stain'),cv2.COLOR_BGR2RGB),caption='Stain-focused view',width='stretch')
            st.dataframe(pd.DataFrame(result['views']).rename_axis('Dataset category'),width='stretch')
            st.caption('The stain-focused view is an automatic colour transformation, not a labelled cancer mask.')


uploaded=st.file_uploader('Upload a microscopy image',type=['jpg','jpeg','png','tif','tiff'])
if uploaded:
    data=uploaded.getvalue()
    image=cv2.imdecode(np.frombuffer(data,dtype=np.uint8),cv2.IMREAD_COLOR)
    if image is None:
        st.error('This image could not be decoded.');st.stop()
    st.image(cv2.cvtColor(image,cv2.COLOR_BGR2RGB),caption=uploaded.name,width='stretch')
    if st.button('Analyze image',type='primary'):
        path=None
        try:
            with tempfile.NamedTemporaryFile(suffix='.png',delete=False) as f:path=Path(f.name)
            if not cv2.imwrite(str(path),image):raise OSError('Could not prepare this image.')
            with st.spinner('Analyzing image...'):
                if mode=='Blood cells + ALL screening':report=(analyze_evidence_combined if experimental else analyze_combined)(path)
                elif mode=='Blood cell analysis':report=analyze(path)
                elif mode=='Leukemia screening (ALL)':report=(screen_evidence if experimental else screen_all)(path)
                else:report=classify_single(path,'wbc')
            report['image_name']=uploaded.name
            st.session_state['analysis']={'key':(uploaded.name,mode,experimental,data),'report':report}
        except Exception as e:
            st.error(f'Analysis could not complete: {e}')
        finally:
            if path is not None:path.unlink(missing_ok=True)
    saved=st.session_state.get('analysis')
    if saved and saved['key']==(uploaded.name,mode,experimental,data):
        report=saved['report']
        if mode=='Blood cells + ALL screening':
            cells,leukemia=st.tabs(['Detected blood cells','ALL screening and review'])
            with cells:show_cells(image,report['cell_analysis'],False)
            with leukemia:show_all(image,report['leukemia_screening'])
        elif mode=='Blood cell analysis':show_cells(image,report,True)
        elif mode=='Leukemia screening (ALL)':show_all(image,report)
        else:
            st.subheader(report['display_label'].replace('_',' ').title())
            st.write(f"Model score: {report['prediction']['score']:.3f}")
            st.write('Status: '+report['status'])
        st.download_button('Download analysis JSON',json.dumps(report,indent=2,allow_nan=False),'blood_cells_and_all.json','application/json')
        with st.expander('Study scope and model details'):
            if 'scope' in report:st.write(report['scope'])
            if 'model_provenance' in report:st.json(report['model_provenance'])
