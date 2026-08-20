import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread(r"C:\Users\ayesh\OneDrive\Desktop\computer vision\computer vision\image.jpeg ")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

gray_float = np.float32(gray)

corners = cv2.cornerHarris(
    gray_float,
    blockSize=2,
    ksize=3,
    k=0.04
)

corners = cv2.dilate(corners, None)

result = img.copy()
result[corners > 0.01 * corners.max()] = [0, 0, 255]

result = cv2.cvtColor(
    result, cv2.COLOR_BGR2RGB
)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(result)
plt.title("Harris Corners")
plt.axis("off")

plt.show()

