import cv2
from matplotlib import pyplot as plt

img = cv2.imread("image.jpeg", 0)
equalized = cv2.equalizeHist(img)
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.imshow(img, cmap='gray')
plt.title("Original Image")
plt.axis('off')
plt.subplot(1,2,2)
plt.imshow(equalized, cmap='gray')
plt.title("Equalized Image")
plt.axis('off')
plt.show()