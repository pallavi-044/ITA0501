import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread(r"C:\Users\ayesh\OneDrive\Desktop\computer vision\computer vision\image.jpeg   ")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

noise = np.random.normal(
    0, 25, gray.shape
).astype(np.uint8)

noisy = cv2.add(gray, noise)

filtered = cv2.GaussianBlur(
    noisy, (5, 5), 0
)


def harris_detection(image):
    gray_float = np.float32(image)

    corners = cv2.cornerHarris(
        gray_float,
        blockSize=2,
        ksize=3,
        k=0.04
    )

    corners = cv2.dilate(corners, None)

    result = cv2.cvtColor(
        image, cv2.COLOR_GRAY2RGB
    )

    result[
        corners > 0.01 * corners.max()
    ] = [255, 0, 0]

    return result


original_corners = harris_detection(gray)
noisy_corners = harris_detection(noisy)
filtered_corners = harris_detection(filtered)

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(gray, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(noisy, cmap="gray")
plt.title("Noisy Image")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(noisy_corners)
plt.title("Harris on Noisy Image")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(filtered_corners)
plt.title("Harris after Noise Removal")
plt.axis("off")

plt.show()

