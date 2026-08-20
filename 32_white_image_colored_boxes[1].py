"""
32. Create a white image of user-defined size and draw four colored boxes
    (Black, Blue, Green, Red) at each corner using OpenCV.

Usage:
    python 32_white_image_colored_boxes.py [width] [height] [box_size]
"""

import cv2
import numpy as np
import sys


def create_image_with_corner_boxes(width=600, height=400, box_size=60):
    # White background (BGR: 255,255,255)
    img = np.full((height, width, 3), 255, dtype=np.uint8)

    b = box_size
    # OpenCV uses BGR color order
    black = (0, 0, 0)
    blue = (255, 0, 0)
    green = (0, 255, 0)
    red = (0, 0, 255)

    # Top-left: Black
    cv2.rectangle(img, (0, 0), (b, b), black, -1)
    # Top-right: Blue
    cv2.rectangle(img, (width - b, 0), (width, b), blue, -1)
    # Bottom-left: Green
    cv2.rectangle(img, (0, height - b), (b, height), green, -1)
    # Bottom-right: Red
    cv2.rectangle(img, (width - b, height - b), (width, height), red, -1)

    cv2.imshow("White Image with Colored Corners", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite("corner_boxes.png", img)
    print("Saved: corner_boxes.png")


if __name__ == "__main__":
    w = int(sys.argv[1]) if len(sys.argv) > 1 else 600
    h = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    box = int(sys.argv[3]) if len(sys.argv) > 3 else 60
    create_image_with_corner_boxes(w, h, box)
