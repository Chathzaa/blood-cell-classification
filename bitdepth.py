import cv2
img = cv2.imread('data/BCCD/train/img/BloodImage_00001.jpeg')  # any real image path
print("dtype:", img.dtype)      # uint8 = 8-bit per channel
print("shape:", img.shape)      # (height, width, channels)
print("min/max pixel value:", img.min(), img.max())  # should be within 0-255 for 8-bit