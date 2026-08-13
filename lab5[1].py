import cv2
from matplotlib import pyplot as plt

img = cv2.imread("image.jpeg")

colors = ('b', 'g', 'r')

for i, color in enumerate(colors):
    hist = cv2.calcHist([img], [i], None, [256], [0,256])
    plt.plot(hist, color=color)

plt.title("Color Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Number of Pixels")
plt.xlim([0,256])
plt.show()