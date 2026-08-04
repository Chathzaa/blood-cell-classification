import hashlib
from pathlib import Path

def md5(p): return hashlib.md5(p.read_bytes()).hexdigest()

for cls in ['Benign','Early','Pre','Pro']:
    train_hashes = {md5(p) for p in Path(f'data/Leukemia_CLS_v2/train/{cls}').glob('*.jpg')}
    val_hashes   = {md5(p) for p in Path(f'data/Leukemia_CLS_v2/val/{cls}').glob('*.jpg')}
    print(cls, "overlap:", len(train_hashes & val_hashes))