# import os
# from pathlib import Path
# import xml.etree.ElementTree as ET
# import cv2
# import shutil

# CLASS_MAP = {'RBC': 0, 'WBC': 1, 'Platelets': 2}


# # ----------------------------
# # Convert XML → YOLO format
# # ----------------------------
# def convert_annotation(xml_path, img_w, img_h):
#     tree = ET.parse(xml_path)
#     root = tree.getroot()

#     lines = []

#     for obj in root.findall('object'):
#         name = obj.find('name').text

#         if name not in CLASS_MAP:
#             continue

#         bb = obj.find('bndbox')

#         xmin = float(bb.find('xmin').text)
#         ymin = float(bb.find('ymin').text)
#         xmax = float(bb.find('xmax').text)
#         ymax = float(bb.find('ymax').text)

#         cx = ((xmin + xmax) / 2) / img_w
#         cy = ((ymin + ymax) / 2) / img_h
#         w = (xmax - xmin) / img_w
#         h = (ymax - ymin) / img_h

#         lines.append(f"{CLASS_MAP[name]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

#     return lines


# # ----------------------------
# # Process one dataset split
# # ----------------------------
# def process_split(bccd_root, out_root, split):

#     img_dir = Path(bccd_root) / split / 'img'
#     ann_dir = Path(bccd_root) / split / 'ann'

#     if not img_dir.exists():
#         print(f"Missing folder: {img_dir}")
#         return

#     images = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.jpeg'))]

#     out_img_dir = Path(out_root) / split / 'images'
#     out_lbl_dir = Path(out_root) / split / 'labels'

#     out_img_dir.mkdir(parents=True, exist_ok=True)
#     out_lbl_dir.mkdir(parents=True, exist_ok=True)

#     for fname in images:

#         img_path = img_dir / fname
#         xml_path = ann_dir / (fname.rsplit('.', 1)[0] + '.xml')

#         img = cv2.imread(str(img_path))

#         if img is None:
#             print("Skipping missing image:", img_path)
#             continue

#         if not xml_path.exists():
#             print("Missing annotation:", xml_path)
#             continue

#         h, w = img.shape[:2]

#         labels = convert_annotation(str(xml_path), w, h)

#         # copy image
#         shutil.copy(str(img_path), str(out_img_dir / fname))

#         # write label file
#         label_file = fname.rsplit('.', 1)[0] + '.txt'

#         with open(out_lbl_dir / label_file, 'w') as f:
#             f.write('\n'.join(labels))


# # ----------------------------
# # Main function
# # ----------------------------
# def prepare_bccd():

#     BASE_DIR = Path(__file__).resolve().parent.parent

#     bccd_root = BASE_DIR / "data" / "BCCD"
#     out_root = BASE_DIR / "data" / "BCCD_YOLO"

#     print("Dataset root:", bccd_root)

#     for split in ['train', 'val', 'test']:
#         print(f"\nProcessing {split}...")
#         process_split(bccd_root, out_root, split)

#     print("\n✅ BCCD dataset prepared successfully!")


# if __name__ == '__main__':
#     prepare_bccd()

import os, json, shutil, cv2

CLASS_MAP = {'RBC': 0, 'WBC': 1, 'Platelets': 2, 'Platelet': 2}

BASE = r'F:\Acdemics\Sem 8\Computer_Vision\Image_Processing_Project'
BCCD   = os.path.join(BASE, 'data', 'BCCD')
OUT    = os.path.join(BASE, 'data', 'BCCD_YOLO')
YAML   = os.path.join(BASE, 'bccd.yaml')

def convert_supervisely_ann(ann_path, img_w, img_h):
    """Convert one Supervisely JSON annotation to YOLO format lines."""
    with open(ann_path, 'r') as f:
        data = json.load(f)

    lines = []
    for obj in data.get('objects', []):
        class_name = obj.get('classTitle', '').strip()
        if class_name not in CLASS_MAP:
            continue

        # Supervisely uses 'points' with 'exterior' [[x1,y1],[x2,y2]]
        pts = obj.get('points', {}).get('exterior', [])
        if len(pts) < 2:
            continue

        x1 = min(p[0] for p in pts)
        y1 = min(p[1] for p in pts)
        x2 = max(p[0] for p in pts)
        y2 = max(p[1] for p in pts)

        cx = ((x1 + x2) / 2) / img_w
        cy = ((y1 + y2) / 2) / img_h
        w  = (x2 - x1) / img_w
        h  = (y2 - y1) / img_h

        # clamp to 0-1
        cx = max(0, min(1, cx))
        cy = max(0, min(1, cy))
        w  = max(0, min(1, w))
        h  = max(0, min(1, h))

        lines.append(
            f"{CLASS_MAP[class_name]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
        )
    return lines

def prepare_bccd():
    total_imgs   = 0
    total_labels = 0
    skipped      = 0

    for split in ['train', 'val', 'test']:
        img_dir = os.path.join(BCCD, split, 'img')
        ann_dir = os.path.join(BCCD, split, 'ann')

        if not os.path.exists(img_dir):
            print(f"  WARNING: {img_dir} not found, skipping {split}")
            continue

        out_img = os.path.join(OUT, split, 'images')
        out_lbl = os.path.join(OUT, split, 'labels')
        os.makedirs(out_img, exist_ok=True)
        os.makedirs(out_lbl, exist_ok=True)

        images = [f for f in os.listdir(img_dir)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        split_imgs = 0
        split_lbls = 0

        for fname in images:
            img_path = os.path.join(img_dir, fname)

            # read image to get dimensions
            img = cv2.imread(img_path)
            if img is None:
                skipped += 1
                continue
            h, w = img.shape[:2]

            # find matching annotation
            # Supervisely ann files are named: imagename.jpg.json
            ann_path = os.path.join(ann_dir, fname + '.json')
            if not os.path.exists(ann_path):
                # try without extension
                base_name = os.path.splitext(fname)[0]
                ann_path = os.path.join(ann_dir, base_name + '.json')

            # copy image
            shutil.copy(img_path, os.path.join(out_img, fname))
            split_imgs += 1

            # convert and save label
            if os.path.exists(ann_path):
                lines = convert_supervisely_ann(ann_path, w, h)
            else:
                lines = []  # empty label if no annotation found

            label_name = os.path.splitext(fname)[0] + '.txt'
            with open(os.path.join(out_lbl, label_name), 'w') as f:
                f.write('\n'.join(lines))
            if lines:
                split_lbls += 1

        print(f"  {split:5s}: {split_imgs} images, {split_lbls} labels converted")
        total_imgs   += split_imgs
        total_labels += split_lbls

    print(f"\nTotal: {total_imgs} images, {total_labels} annotated")
    if skipped:
        print(f"Skipped (unreadable): {skipped}")

    # write bccd.yaml
    yaml_content = f"""path: {OUT}
train: train/images
val: val/images
test: test/images
nc: 3
names: ['RBC', 'WBC', 'Platelet']
"""
    with open(YAML, 'w') as f:
        f.write(yaml_content)
    print(f"\nbccd.yaml written to: {YAML}")
    print("\nAll done! BCCD_YOLO is ready for training.")

if __name__ == '__main__':
    # first print what we find in ann folder so you can verify
    sample_ann = os.path.join(BCCD, 'train', 'ann')
    if os.path.exists(sample_ann):
        sample_files = os.listdir(sample_ann)[:3]
        print("Sample annotation files found:")
        for sf in sample_files:
            print(f"  {sf}")
        # print contents of first one
        if sample_files:
            with open(os.path.join(sample_ann, sample_files[0])) as f:
                import json
                data = json.load(f)
                print(f"\nFirst annotation file preview:")
                print(f"  Keys: {list(data.keys())}")
                if 'objects' in data:
                    print(f"  Number of objects: {len(data['objects'])}")
                    if data['objects']:
                        print(f"  First object keys: {list(data['objects'][0].keys())}")
                        print(f"  Class: {data['objects'][0].get('classTitle')}")
        print()

    prepare_bccd()

import os, shutil, random, cv2

BASE = r'F:\Acdemics\Sem 8\Computer_Vision\Project'

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


# ── LEUKEMIA DATASET ──────────────────────────────────────────────────
def prepare_leukemia():
    src = os.path.join(BASE, r'data\Leukemia\Original')
    out = os.path.join(BASE, r'data\Leukemia_YOLO')

    # Benign=0 (normal), Early/Pre/Pro=1 (leukemia/blast)
    CLASS_MAP = {'Benign': 0, 'Early': 1, 'Pre': 1, 'Pro': 1}

    all_images = []
    for class_name, class_id in CLASS_MAP.items():
        class_dir = os.path.join(src, class_name)
        if not os.path.exists(class_dir):
            print(f"  NOT FOUND: {class_dir}")
            continue
        imgs = [f for f in os.listdir(class_dir)
                if f.lower().endswith(('.jpg','.jpeg','.png'))]
        for img in imgs:
            all_images.append((os.path.join(class_dir, img), class_id, class_name, img))

    random.shuffle(all_images)
    n = len(all_images)
    splits = {
        'train': all_images[:int(n*0.70)],
        'val':   all_images[int(n*0.70):int(n*0.85)],
        'test':  all_images[int(n*0.85):]
    }

    for split_name, items in splits.items():
        img_out = os.path.join(out, split_name, 'images')
        lbl_out = os.path.join(out, split_name, 'labels')
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)

        for img_path, class_id, class_name, fname in items:
            img = cv2.imread(img_path)
            if img is None:
                continue
            new_name = f"{class_name}_{fname}"
            shutil.copy(img_path, os.path.join(img_out, new_name))
            label_name = os.path.splitext(new_name)[0] + '.txt'
            with open(os.path.join(lbl_out, label_name), 'w') as f:
                f.write(f"{class_id} 0.5 0.5 0.9 0.9\n")

        print(f"  Leukemia {split_name}: {len(items)} images")

    yaml_path = os.path.join(BASE, 'leukemia.yaml')
    with open(yaml_path, 'w') as f:
        f.write(f"""path: {out}
train: train/images
val: val/images
test: test/images
nc: 2
names: ['Normal', 'Blast']
""")
    print(f"  leukemia.yaml written!")


if __name__ == '__main__':
    print("Preparing BloodCell Images dataset...")
    prepare_bloodcell()
    print("\nPreparing Leukemia dataset...")
    prepare_leukemia()
    print("\nAll datasets ready!")
