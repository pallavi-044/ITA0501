import cv2
import matplotlib.pyplot as plt

img = cv2.imread(r"C:\Users\ayesh\OneDrive\Desktop\computer vision\computer vision\image.jpeg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sobel_x = cv2.Sobel(
    gray, cv2.CV_64F, 1, 0, ksize=3
)
sobel_y = cv2.Sobel(
    gray, cv2.CV_64F, 0, 1, ksize=3
)

sobel = cv2.magnitude(sobel_x, sobel_y)
sobel = cv2.convertScaleAbs(sobel)

canny = cv2.Canny(gray, 100, 200)

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(gray, cmap="gray")
plt.title("Original")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(sobel, cmap="gray")
plt.title("Sobel Edge")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(canny, cmap="gray")
plt.title("Canny Edge")
plt.axis("off")

plt.show()
