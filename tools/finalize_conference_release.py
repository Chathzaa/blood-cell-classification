"""Verify and package the final conference manuscript without altering experiments."""
from pathlib import Path
import argparse, hashlib, json, re, shutil, zipfile
from release_manifest import release_files
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / 'paper/final_conference'
DEST = ROOT / 'deliverables'
CHECK = ROOT / 'audit/final_package_check'
FAMILIES = ['original', 'context_augmentation', 'full_context_augmentation', 'amer', 'unmatched_amer']
NAMES = ['Original', 'Matched augmentation', 'Full context augmentation', 'AMER', 'Unmatched AMER']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def pdf_text(path):
    return [p.extract_text() for p in PdfReader(path).pages]


def verify_paper():
    previous = read(ROOT / 'audit/evidence_release_verification.json')
    assert previous['status'] == 'passed'
    assert previous['study_sha256'] == sha(ROOT / 'outputs/all_novel/study.json')
    assert previous['all_metrics_recomputed'] and previous['validation_only_selection_verified']
    source = (PAPER / 'main.tex').read_text(encoding='utf-8')
    refs = (PAPER / 'references.tex').read_text(encoding='utf-8')
    keys = re.findall(r'\\bibitem\{([^}]+)\}', refs)
    cited = [k for group in re.findall(r'\\cite\{([^}]+)\}', source) for k in group.split(',')]
    assert len(keys) == len(set(keys)) == 31
    assert set(cited) == set(keys)
    assert list(dict.fromkeys(cited)) == keys
    assert source.count(r'\begin{figure*}[t]') == 3
    assert source.count(r'\includegraphics[width=.98\textwidth]') == 3
    assert len(re.findall(r'\\begin\{table\}\[![tb]\]', source)) == 6
    assert 'declarations' not in source.lower()
    assert r'\linebreakand' in source
    assert 'The authors thank Rangana W. G. for supervising this project.' in source
    assert '—' not in source and '–' not in source
    pages = pdf_text(PAPER / 'main.pdf')
    assert len(pages) == 6 and 'We contribute' in pages[0]
    text = re.sub(r'\s+', ' ', '\n'.join(pages))
    assert '??' not in text
    for phrase in ['previously evaluated', 'Patient identifiers are unavailable', 'did not outperform', 'exploratory']:
        assert phrase in text, phrase
    log = (PAPER / 'main.log').read_text(errors='replace')
    assert 'Overfull' not in log and 'undefined references' not in log
    assert not re.search(r'(Citation|Reference).*undefined', log)
    study = read(ROOT / 'outputs/all_novel/study.json')
    table = (PAPER / 'tables.tex').read_text()
    for family, name in zip(FAMILIES, NAMES):
        e = study['evaluation']['none'][family]
        metrics = f"{name} & {e['macro_f1']:.3f} & {e['mcc']:.3f} & {100*e['binary']['sensitivity']:.1f} & {100*e['binary']['specificity']:.1f}"
        assert metrics in table
        accuracy = name + ' & ' + ' & '.join(f"{100*study['evaluation'][c][family]['accuracy']:.1f}" for c in ['none','blur','colour','context_colour'])
        assert accuracy in table
    for condition, label in [('none','Original'),('blur','Blur'),('colour','Global colour'),('context_colour','Context colour')]:
        parts=[]
        for baseline, key in [('original','amer_minus_original'),('context_augmentation','amer_minus_context_augmentation')]:
            d=100*(study['evaluation'][condition]['amer']['accuracy']-study['evaluation'][condition][baseline]['accuracy'])
            lo,hi=study['bootstrap'][condition][key]
            parts.append(f'{d:+.1f} [{100*lo:+.1f}, {100*hi:+.1f}]')
        assert label+' & '+' & '.join(parts) in table
    return dict(status='passed', pages=6, references=31, figures=3, tables=6,
                contribution_page=1, results_checked_against_recorded_study=True,
                experiment_verification=previous, unresolved_references=False,
                overfull_boxes=False, prior_test_use_disclosed=True,
                scope='Blood cell detection and image-level ALL screening',
                claim='ALL-specific supervised evidence formulation and controlled empirical evaluation; no superiority claim',
                pdf_sha256=sha(PAPER/'main.pdf'))


def package():
    report=verify_paper()
    (ROOT/'audit/final_conference_verification.json').write_text(json.dumps(report,indent=2))
    pdf=DEST/'Final_Conference_Paper.pdf'
    shutil.copy2(PAPER/'main.pdf',pdf)
    inputs=list(PAPER.glob('*.tex'))+[PAPER/'BUILD.md']+list((PAPER/'figures').glob('*.pdf'))
    latex=DEST/'Final_Conference_LaTeX.zip'
    with zipfile.ZipFile(latex,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(inputs): z.write(p,p.relative_to(PAPER).as_posix())
    CHECK.mkdir(exist_ok=True)
    with zipfile.ZipFile(latex) as z:
        assert z.testzip() is None
        for name in z.namelist():
            assert (CHECK/name).resolve().is_relative_to(CHECK.resolve())
        z.extractall(CHECK)
    shutil.copy2(ROOT/'FINAL_DELIVERY.md',DEST/'START_HERE.md')
    project=DEST/'Final_Blood_Cell_Project.zip'
    with zipfile.ZipFile(project,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in release_files(): z.write(p,p.relative_to(ROOT).as_posix())
    for path in [latex,project]:
        with zipfile.ZipFile(path) as z: assert z.testzip() is None
    manifest={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in [pdf,latex,project]}
    (DEST/'FINAL_SHA256.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(dict(pages=6,references=31,artifacts=manifest,independent_build_directory=str(CHECK)),indent=2))


def check_build():
    verify_paper()
    assert pdf_text(CHECK/'main.pdf')==pdf_text(PAPER/'main.pdf')
    log=(CHECK/'main.log').read_text(errors='replace')
    assert 'Overfull' not in log and 'undefined references' not in log
    manifest=read(DEST/'FINAL_SHA256.json')
    for name,record in manifest.items(): assert sha(DEST/name)==record['sha256']
    with zipfile.ZipFile(DEST/'Final_Blood_Cell_Project.zip') as z:
        assert z.read('paper/final_conference/main.pdf')==(DEST/'Final_Conference_Paper.pdf').read_bytes()
        assert z.read('paper/final_conference/main.tex')==(PAPER/'main.tex').read_bytes()
    report=dict(status='passed',pages=6,references=31,independent_latex_build=True,
                pdf_text_matches=True,project_pdf_matches=True,zip_checksums_verified=True,
                pdf_sha256=sha(DEST/'Final_Conference_Paper.pdf'))
    (DEST/'FINAL_BUILD_VERIFIED.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--check-build',action='store_true');args=parser.parse_args()
    check_build() if args.check_build else package()
