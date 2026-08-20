import cv2
import matplotlib.pyplot as plt

img = cv2.imread(r"C:\Users\ayesh\OneDrive\Desktop\computer vision\computer vision\image.jpeg")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

_, binary = cv2.threshold(
    gray, 127, 255, cv2.THRESH_BINARY
)

edges_gray = cv2.Canny(gray, 100, 200)
edges_binary = cv2.Canny(binary, 100, 200)

plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Original RGB")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(gray, cmap="gray")
plt.title("Grayscale")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(binary, cmap="gray")
plt.title("Binary")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(edges_gray, cmap="gray")
plt.title("Grayscale Edges")
plt.axis("off")

plt.show()

