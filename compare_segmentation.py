import sys
sys.path.append('src')
import cv2, numpy as np, matplotlib.pyplot as plt
from preprocessing import preprocess_image
from segmentation import (otsu_threshold, morphological_cleanup,
                          distance_transform, watershed_segment,
                          filter_contours, contours_from_markers)

img_path = 'data/BCCD_YOLO/val/images/BloodImage_00002.jpeg'
proc = preprocess_image(img_path)

binary  = otsu_threshold(proc)
clean   = morphological_cleanup(binary)
dist    = distance_transform(clean)
markers = watershed_segment(proc, clean, dist)

old = filter_contours(clean)                 # OLD behaviour (no watershed)
new = contours_from_markers(markers)         # NEW behaviour (watershed used)

a, b = proc.copy(), proc.copy()
cv2.drawContours(a, old, -1, (0, 255, 0), 2)
cv2.drawContours(b, new, -1, (0, 255, 0), 2)

fig, ax = plt.subplots(1, 2, figsize=(14, 6))
ax[0].imshow(cv2.cvtColor(a, cv2.COLOR_BGR2RGB))
ax[0].set_title(f'BEFORE: Otsu only - {len(old)} regions\n(clumps merged)')
ax[1].imshow(cv2.cvtColor(b, cv2.COLOR_BGR2RGB))
ax[1].set_title(f'AFTER: watershed - {len(new)} regions\n(clumps separated)')
for x in ax: x.axis('off')
plt.tight_layout()
plt.savefig('outputs/watershed_before_after.png', dpi=200)
plt.show()
