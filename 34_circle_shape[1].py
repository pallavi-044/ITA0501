"""
34. Create a Circle shape using OpenCV.

Usage:
    python 34_circle_shape.py
"""

import cv2
import numpy as np

if __name__ == "__main__":
    img = np.full((400, 600, 3), 255, dtype=np.uint8)

    center = (300, 200)
    radius = 100
    color = (255, 0, 0)   # Blue (BGR)
    thickness = 3          # use -1 to fill the circle

    cv2.circle(img, center, radius, color, thickness)

    cv2.imshow("Circle", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite("circle.png", img)
    print("Saved: circle.png")
