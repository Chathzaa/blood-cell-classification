import random, shutil, hashlib
from pathlib import Path

random.seed(42)
CLASSES = ['Benign', 'Early', 'Pre', 'Pro']
SRC = Path('data/Leukemia/Original')
OUT = Path('data/Leukemia_CLS_v2')

def md5(p):
    return hashlib.md5(p.read_bytes()).hexdigest()

for cls in CLASSES:
    files = list((SRC/cls).glob('*.jpg'))

    # Dedupe by content hash - keep only the first file for each unique image
    seen_hashes = {}
    for f in files:
        h = md5(f)
        if h not in seen_hashes:
            seen_hashes[h] = f
    unique_files = list(seen_hashes.values())

    dropped = len(files) - len(unique_files)
    if dropped:
        print(f"{cls}: dropped {dropped} exact-duplicate file(s) before split")

    random.shuffle(unique_files)
    n_val = int(0.2 * len(unique_files))
    val_files, train_files = unique_files[:n_val], unique_files[n_val:]

    for split_name, flist in [('train', train_files), ('val', val_files)]:
        out_dir = OUT/split_name/cls
        # wipe old split output for this class so no stale duplicates linger
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for f in flist:
            shutil.copy(f, out_dir/f.name)

    print(f"{cls}: {len(unique_files)} unique -> {len(train_files)} train / {len(val_files)} val")