"""
33. Create a Rectangle shape using OpenCV.

Usage:
    python 33_rectangle_shape.py
"""

import cv2
import numpy as np

if __name__ == "__main__":
    img = np.full((400, 600, 3), 255, dtype=np.uint8)

    # Rectangle from (x1, y1) top-left to (x2, y2) bottom-right
    top_left = (150, 100)
    bottom_right = (450, 300)
    color = (0, 0, 255)   # Red (BGR)
    thickness = 3          # use -1 to fill the rectangle

    cv2.rectangle(img, top_left, bottom_right, color, thickness)

    cv2.imshow("Rectangle", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite("rectangle.png", img)
    print("Saved: rectangle.png")
