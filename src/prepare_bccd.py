import os, json, shutil, cv2

CLASS_MAP = {'RBC': 0, 'WBC': 1, 'Platelets': 2, 'Platelet': 2}

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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