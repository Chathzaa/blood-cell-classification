import os, shutil, random, cv2

random.seed(42)          

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
    prepare_leukemia()

