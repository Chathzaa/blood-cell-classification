import cv2, numpy as np, matplotlib.pyplot as plt
import os

image_path = "data/BloodCell_YOLO/train/images/LYMPHOCYTE__1_17.jpeg"

print(os.path.abspath(image_path))
print(os.path.exists(image_path))
# Point this at any single WBC image from the BloodCell dataset
img_path = 'data/BloodCell_YOLO/train/images/LYMPHOCYTE__1_17.jpeg'

roi = cv2.imread(img_path)
if roi is None:
    raise SystemExit(f"Could not read {img_path} - fix the path")

b, g, r = cv2.split(roi)
hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
s = hsv[:, :, 1]

candidates = {
    'blue INV (current code)': (b, cv2.THRESH_BINARY_INV),
    'green INV': (g, cv2.THRESH_BINARY_INV),
    'saturation (normal)': (s, cv2.THRESH_BINARY),
}

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
axes[0].imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
axes[0].set_title('Original')
axes[0].axis('off')

for ax, (name, (chan, mode)) in zip(axes[1:], candidates.items()):
    _, mask = cv2.threshold(chan, 0, 255, mode + cv2.THRESH_OTSU)
    pct = 100 * np.count_nonzero(mask) / mask.size
    ax.imshow(mask, cmap='gray')
    ax.set_title(f'{name}\n{pct:.0f}% white')
    ax.axis('off')

plt.tight_layout()
plt.savefig('outputs/nucleus_channel_test.png', dpi=150)
plt.show()
