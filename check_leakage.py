import re
from pathlib import Path

CLASSES = ['EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL']
ROOT = Path('data/BloodCell_CLS_v2')

def source_id(stem):
    m = re.search(r'_\d+_(\d+)$', stem)
    return m.group(1) if m else stem

total_overlap = 0
for cls in CLASSES:
    train = {source_id(p.stem) for p in (ROOT/'train'/cls).glob('*.jpeg')}
    val   = {source_id(p.stem) for p in (ROOT/'val'/cls).glob('*.jpeg')}
    overlap = train & val
    total_overlap += len(overlap)
    print(f"{cls}: train sources={len(train)} val sources={len(val)} overlap={len(overlap)}")

print("\nTOTAL OVERLAP:", total_overlap, "<-- must be 0")