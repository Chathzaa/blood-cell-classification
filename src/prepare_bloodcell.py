import os, shutil, random, cv2

random.seed(42)          

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ── BLOODCELL IMAGES (WBC subtypes) ───────────────────────────────────
def prepare_bloodcell():
    src  = os.path.join(BASE, r'data\BloodCellImages\dataset2-master\dataset2-master\images')
    out  = os.path.join(BASE, r'data\BloodCell_YOLO')

    # class map — folder name → class id
    CLASS_MAP = {'EOSINOPHIL': 0, 'LYMPHOCYTE': 1, 'MONOCYTE': 2, 'NEUTROPHIL': 3}

    splits = {'train': 'TRAIN', 'test': 'TEST'}

    for out_split, src_split in splits.items():
        for class_name, class_id in CLASS_MAP.items():
            src_dir = os.path.join(src, src_split, class_name)
            if not os.path.exists(src_dir):
                print(f"  NOT FOUND: {src_dir}")
                continue

            images = [f for f in os.listdir(src_dir)
                      if f.lower().endswith(('.jpg','.jpeg','.png'))]

            # for train: 85% train, 15% val
            if out_split == 'train':
                random.shuffle(images)
                val_imgs   = images[:int(len(images)*0.15)]
                train_imgs = images[int(len(images)*0.15):]
                batches = [('train', train_imgs), ('val', val_imgs)]
            else:
                batches = [('test', images)]

            for dest_split, imgs in batches:
                img_out = os.path.join(out, dest_split, 'images')
                lbl_out = os.path.join(out, dest_split, 'labels')
                os.makedirs(img_out, exist_ok=True)
                os.makedirs(lbl_out, exist_ok=True)

                for fname in imgs:
                    img_path = os.path.join(src_dir, fname)
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                    h, w = img.shape[:2]

                    # whole image bounding box (single cell images)
                    cx, cy = 0.5, 0.5
                    bw, bh = 0.9, 0.9

                    new_name = f"{class_name}_{fname}"
                    shutil.copy(img_path, os.path.join(img_out, new_name))
                    label_name = os.path.splitext(new_name)[0] + '.txt'
                    with open(os.path.join(lbl_out, label_name), 'w') as f:
                        f.write(f"{class_id} {cx} {cy} {bw} {bh}\n")

    # count results
    for sp in ['train','val','test']:
        img_dir = os.path.join(out, sp, 'images')
        if os.path.exists(img_dir):
            print(f"  BloodCell {sp}: {len(os.listdir(img_dir))} images")

    # write yaml
    yaml_path = os.path.join(BASE, 'bloodcell.yaml')
    with open(yaml_path, 'w') as f:
        f.write(f"""path: {out}
train: train/images
val: val/images
test: test/images
nc: 4
names: ['Eosinophil', 'Lymphocyte', 'Monocyte', 'Neutrophil']
""")
    print(f"  bloodcell.yaml written!")

if __name__ == '__main__':
    prepare_bloodcell()

