import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread(r"C:\Users\ayesh\OneDrive\Desktop\computer vision\computer vision\image.jpeg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

noise = np.random.normal(0, 25, gray.shape).astype(np.uint8)
noisy = cv2.add(gray, noise)

filtered = cv2.GaussianBlur(noisy, (5, 5), 0)

normalized = cv2.normalize(
    filtered, None, 0, 255, cv2.NORM_MINMAX
)

edges_noisy = cv2.Canny(noisy, 100, 200)
edges_processed = cv2.Canny(normalized, 100, 200)

plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.imshow(gray, cmap="gray")
plt.title("Original")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(noisy, cmap="gray")
plt.title("Noisy")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(filtered, cmap="gray")
plt.title("Filtered")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.imshow(normalized, cmap="gray")
plt.title("Normalized")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.imshow(edges_noisy, cmap="gray")
plt.title("Edges - Noisy")
plt.axis("off")

plt.subplot(2, 3, 6)
plt.imshow(edges_processed, cmap="gray")
plt.title("Edges - Processed")
plt.axis("off")

plt.show()
