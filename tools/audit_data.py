"""Read-only audit of supplied data, with file/pixel hashes and source provenance."""
from pathlib import Path
from collections import Counter, defaultdict
import csv
import hashlib
import json
import re
import cv2

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'audit'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_split(name):
    base = ROOT / 'data' / name
    groups = {'file': defaultdict(list), 'pixels': defaultdict(list)}
    rows = []
    for path in sorted(base.rglob('*')):
        if path.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
            continue
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f'Unreadable image: {path}')
        rel = path.relative_to(base)
        row = dict(dataset=name, path=path.relative_to(ROOT).as_posix(), split=rel.parts[0],
                   label=rel.parts[1], sha256=sha(path),
                   pixel_sha256=hashlib.sha256(str(image.shape).encode()+image.tobytes()).hexdigest())
        rows.append(row)
        groups['file'][row['sha256']].append(row)
        groups['pixels'][row['pixel_sha256']].append(row)
    summary = {'counts': dict(Counter(r['split'] for r in rows))}
    for kind, index in groups.items():
        duplicates = [g for g in index.values() if len(g) > 1]
        cross = [g for g in duplicates if len({r['split'] for r in g}) > 1]
        summary[kind] = dict(duplicate_groups=len(duplicates), cross_split_groups=len(cross),
                             cross_split_files=sum(map(len, cross)))
        (OUT / f'{name}_{kind}_duplicates.json').write_text(json.dumps(cross, indent=2))
    print(name, summary, flush=True)
    return rows, summary


def audit_augmented_sources(rows):
    origin = ROOT / 'data/BloodCellImages/dataset2-master/dataset2-master/images'
    lookup = defaultdict(list)
    for split in ('TRAIN', 'TEST'):
        for p in sorted((origin / split).rglob('*.jpeg')):
            m = re.fullmatch(r'_(\d+)_(\d+)', p.stem)
            if not m:
                raise ValueError(p)
            # Keras saves {prefix}_{index}_{random hash}; namespaces restart by folder.
            group = f'{split}/{p.parent.name}/{m.group(1)}'
            lookup[(p.parent.name, sha(p))].append((group, p.relative_to(ROOT).as_posix()))
    groups = defaultdict(set)
    mapped = []
    ambiguous = 0
    for r in rows:
        matches = lookup.get((r['label'], r['sha256']), [])
        candidates = sorted({m[0] for m in matches})
        if len(candidates) != 1:
            ambiguous += 1
        for g in candidates:
            groups[g].add(r['split'])
        mapped.append(dict(r, source_groups=';'.join(candidates)))
    crossed = {g: sorted(s) for g, s in groups.items() if len(s) > 1}
    affected = sum(any(g in crossed for g in r['source_groups'].split(';')) for r in mapped if r['split']=='val')
    summary = dict(original_files=sum(map(len, lookup.values())), mapped_files=len(mapped),
                   ambiguous_files=ambiguous, source_groups=len(groups),
                   cross_split_source_groups=len(crossed), validation_files_in_crossed_groups=affected)
    with (OUT / 'wbc_augmentation_provenance.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(mapped[0])); w.writeheader(); w.writerows(mapped)
    (OUT / 'wbc_crossed_source_groups.json').write_text(json.dumps(crossed, indent=2))
    print('WBC provenance', summary, flush=True)
    return summary


def main():
    OUT.mkdir(exist_ok=True)
    all_rows, summary = [], {}
    for name in ('BCCD_YOLO', 'BloodCell_CLS_v2', 'Leukemia_CLS_v2'):
        rows, summary[name] = audit_split(name)
        all_rows += rows
        if name == 'BloodCell_CLS_v2':
            summary['wbc_provenance'] = audit_augmented_sources(rows)
    with (OUT / 'file_manifest.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0])); w.writeheader(); w.writerows(all_rows)
    original = list((ROOT/'data/Leukemia/Original').rglob('*.jpg'))
    hashes = Counter(sha(p) for p in original)
    summary['leukemia_original'] = dict(files=len(original), unique_file_hashes=len(hashes),
                                       redundant_files=len(original)-len(hashes), patient_ids_available=False)
    (OUT/'data_audit.json').write_text(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
