import re, random, shutil
from pathlib import Path

random.seed(42)
CLASSES = ['EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL']
SRC_ROOT = Path('data/BloodCellImages/dataset2-master/dataset2-master/images')
OUT = Path('data/BloodCell_CLS_v2')  # classification folder-per-class output

def source_id(stem):
    m = re.search(r'_\d+_(\d+)$', stem)
    return m.group(1) if m else stem

for cls in CLASSES:
    # pool BOTH original TRAIN and TEST folders for this class together
    files = list((SRC_ROOT/'TRAIN'/cls).glob('*.jpeg')) + list((SRC_ROOT/'TEST'/cls).glob('*.jpeg'))
    groups = {}
    for f in files:
        groups.setdefault(source_id(f.stem), []).append(f)

    ids = list(groups.keys())
    random.shuffle(ids)
    split = int(0.8 * len(ids))
    train_ids, val_ids = ids[:split], ids[split:]

    for split_name, id_list in [('train', train_ids), ('val', val_ids)]:
        out_dir = OUT/split_name/cls
        out_dir.mkdir(parents=True, exist_ok=True)
        for i in id_list:
            for f in groups[i]:
                shutil.copy(f, out_dir/f.name)

    print(f"{cls}: {len(ids)} unique sources -> {len(train_ids)} train / {len(val_ids)} val "
          f"({sum(len(groups[i]) for i in train_ids)} train imgs, {sum(len(groups[i]) for i in val_ids)} val imgs)")