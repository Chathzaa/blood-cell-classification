"""Reproducible tables and publication figures for the evidence-training study."""
from pathlib import Path
import csv,json,re,sys
import cv2,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path.insert(0,str(ROOT))
from src.evidence_regularization import intervention_views
from src.all_views import LABELS
plt.rcParams.update({'font.family':'Times New Roman','font.size':9,'pdf.fonttype':42,
                     'axes.spines.top':False,'axes.spines.right':False})
FAMILIES=['original','context_augmentation','full_context_augmentation','amer','unmatched_amer']
NAMES=['Original','Matched augmentation','Full context augmentation','AMER','Unmatched AMER']


def savefig(fig,name):
    fig.savefig(HERE/'figures'/f'{name}.pdf',bbox_inches='tight',pad_inches=.04)
    fig.savefig(HERE/'figures'/f'{name}.png',bbox_inches='tight',pad_inches=.04,dpi=170)
    plt.close(fig)


def references():
    raw=(HERE/'references.tex').read_text(encoding='utf-8')
    entries=re.findall(r'\\bibitem\{([^}]+)\}(.*?)(?=\\bibitem|\\end\{thebibliography\})',raw,re.S)
    kept={k:v.strip() for k,v in entries}
    source=(HERE/'main.tex').read_text(encoding='utf-8');order=[]
    for group in re.findall(r'\\cite\{([^}]+)\}',source):
        for k in group.split(','):
            if k not in order:order.append(k)
    assert set(order)==set(kept),(set(kept)-set(order),set(order)-set(kept))
    (HERE/'references.tex').write_text('\\begin{thebibliography}{99}\n'+'\n\n'.join('\\bibitem{'+k+'} '+kept[k] for k in order)+'\n\\end{thebibliography}\n',encoding='utf-8')


def results(study):
    defs={};tables={};ev=study['evaluation']
    for key,name in [('original','Original'),('context_augmentation','Aug'),('amer','Amer'),('unmatched_amer','Unmatched')]:
        for condition,prefix in [('none','Clean'),('blur','Blur'),('colour','Colour'),('context_colour','Context')]:
            defs[name+prefix]=f"{100*ev[condition][key]['accuracy']:.1f}"
    binary=ev['none']['amer']['binary']
    for key in ('tn','fp','fn','tp'):defs['Amer'+key.upper()]=str(binary[key])
    defs['AmerSensitivity']=f"{100*binary['sensitivity']:.1f}";defs['AmerSpecificity']=f"{100*binary['specificity']:.1f}"
    defs['AmerFone']=f"{ev['none']['amer']['macro_f1']:.3f}"
    defs['AmerChanged']=f"{100*study['context_prediction_change']['amer']:.1f}"
    defs['OriginalChanged']=f"{100*study['context_prediction_change']['original']:.1f}"
    tables['AccuracyRows']=[name+' & '+' & '.join(f"{100*ev[c][k]['accuracy']:.1f}" for c in ('none','blur','colour','context_colour'))+r' \\' for k,name in zip(FAMILIES,NAMES)]
    tables['CleanRows']=[name+f" & {ev['none'][k]['macro_f1']:.3f} & {ev['none'][k]['mcc']:.3f} & {100*ev['none'][k]['binary']['sensitivity']:.1f} & {100*ev['none'][k]['binary']['specificity']:.1f}"+r' \\' for k,name in zip(FAMILIES,NAMES)]
    tables['DifferenceRows']=[]
    for c,name in [('none','Original'),('blur','Blur'),('colour','Global colour'),('context_colour','Context colour')]:
        pieces=[]
        for key,baseline in [('amer_minus_original','original'),('amer_minus_context_augmentation','context_augmentation')]:
            d=100*(ev[c]['amer']['accuracy']-ev[c][baseline]['accuracy']);lo,hi=np.array(study['bootstrap'][c][key])*100
            pieces.append(f'{d:+.1f} [{lo:+.1f}, {hi:+.1f}]')
        tables['DifferenceRows'].append(name+' & '+' & '.join(pieces)+r' \\')
    sel=study['selection']['families']
    tables['SelectionRows']=[]
    for k,name in zip(FAMILIES,NAMES):
        s=sel[k];p=s['candidates'][s['selected']]
        tables['SelectionRows'].append(name+f" & {p['l2']:g} & {p['ranking_weight']:g} & {p['macro_f1']:.3f}"+r' \\')
    (HERE/'numbers.tex').write_text('\n'.join('\\newcommand{\\'+k+'}{'+v+'}' for k,v in defs.items())+'\n')
    (HERE/'tables.tex').write_text('\n'.join('\\newcommand{\\'+k+'}{%\n'+'\n'.join(v)+'%\n}' for k,v in tables.items())+'\n')


def architecture():
    fig,ax=plt.subplots(figsize=(7.0,2.2));ax.set(xlim=(0,12),ylim=(-.5,3.6));ax.axis('off')
    def box(x,y,w,h,t,color='#15546b',fill='#eff8fa'):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.035',edgecolor=color,facecolor=fill,lw=.9))
        ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=8.5,color=color)
    def arrow(a,b):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=8,color='#15546b',lw=.9))
    ax.text(0,3.38,'TRAINING',fontweight='bold',fontsize=9,color='#9b5b11')
    box(.05,1.95,1.7,.9,'Original image\nand category')
    box(2.15,1.95,2.2,.9,'Automatic stain\nproposal + context')
    box(4.75,1.95,2.05,.9,'Original + two\nequal area erasures')
    box(7.2,1.95,1.95,.9,'Frozen CNN\nrepresentations')
    box(9.55,1.8,2.35,1.2,'Classification loss\n+ relative evidence\nranking penalty','#9b5b11','#fff7e9')
    for a,b in [((1.8,2.4),(2.1,2.4)),((4.4,2.4),(4.7,2.4)),((6.85,2.4),(7.15,2.4)),((9.2,2.4),(9.5,2.4))]:arrow(a,b)
    ax.text(0,1.25,'INFERENCE',fontweight='bold',fontsize=9,color='#15546b')
    box(.05,-.03,1.7,.85,'New image')
    box(2.8,.58,4.7,.55,'Cell detector: RBC, WBC, platelet')
    box(2.8,-.35,4.7,.55,'One CNN pass + trained ALL head')
    box(9.1,-.03,2.75,.85,'Counts + separate\nALL screening')
    arrow((1.8,.65),(2.75,.85));arrow((1.8,.12),(2.75,-.08))
    arrow((7.55,.85),(9.05,.65));arrow((7.55,-.08),(9.05,.12))
    # Outputs share a report; detector crops are not used as leukemia labels.
    savefig(fig,'architecture')


def interventions():
    rows=list(csv.DictReader((ROOT/'data/ALL_verified/manifest.csv').open()))
    row=next(r for r in rows if r['split']=='train' and r['label']=='Benign')
    v,m=intervention_views(cv2.imread(str(ROOT/row['path'])))
    fig,axes=plt.subplots(1,3,figsize=(7.0,2.22))
    for ax,k,title in zip(axes,['original','context','region'],['(a) Original field','(b) Context erasure','(c) Stain region erasure']):
        ax.imshow(v[k][:,:,::-1]);ax.axis('off');ax.set_title(title,fontsize=9)
    fig.tight_layout(pad=.1,w_pad=1.2);savefig(fig,'interventions')


def findings(study):
    fig,(ax,bx)=plt.subplots(1,2,figsize=(7.0,2.55),gridspec_kw={'width_ratios':[1,1.3]})
    cm=np.array(study['evaluation']['none']['amer']['confusion']);ax.imshow(cm/cm.sum(axis=1)[:,None],vmin=0,vmax=1,cmap='Blues')
    for i in range(4):
        for j in range(4):ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]/cm[i].sum()>.55 else '#173348',fontsize=9)
    ax.set(xticks=range(4),yticks=range(4),xticklabels=LABELS,yticklabels=LABELS,xlabel='Predicted category',ylabel='Dataset category')
    ax.set_title('(a) AMER, original test images',fontsize=9)
    conditions=['none','blur','colour','context_colour'];names=['Original','Blur','Global colour','Context colour']
    for offset,key,label,color in [(-.10,'amer_minus_original','vs original head','#176f8a'),(.10,'amer_minus_context_augmentation','vs matched augmentation','#c07920')]:
        ds=[];lower=[];upper=[]
        base='original' if key=='amer_minus_original' else 'context_augmentation'
        for c in conditions:
            d=100*(study['evaluation'][c]['amer']['accuracy']-study['evaluation'][c][base]['accuracy'])
            lo,hi=np.array(study['bootstrap'][c][key])*100;ds.append(d);lower.append(d-lo);upper.append(hi-d)
        bx.errorbar(ds,np.arange(4)+offset,xerr=[lower,upper],fmt='o',ms=3.5,capsize=2,label=label,color=color)
    bx.axvline(0,color='.5',lw=.8,ls=':');bx.set(yticks=range(4),yticklabels=names,xlabel='Accuracy difference (percentage points)')
    bx.invert_yaxis();bx.set_title('(b) Paired group bootstrap, 95% intervals',fontsize=9)
    bx.legend(fontsize=7,loc='best',frameon=False);fig.tight_layout(pad=.7,w_pad=2.4);savefig(fig,'findings')


def main():
    study=json.loads((ROOT/'outputs/all_novel/study.json').read_text())
    results(study);references();architecture();interventions();findings(study)
    print('Evidence paper assets generated from all five recorded model families.')


if __name__=='__main__':main()
