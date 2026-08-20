"""
35. Display a user-entered text string on an image using OpenCV.

Usage:
    python 35_display_text.py [image_path]
(If no image path is given, a blank white canvas is used.)
"""

import cv2
import numpy as np
import sys


def put_text_on_image(img):
    text = input("Enter the text to display on the image: ")

    font = cv2.FONT_HERSHEY_SIMPLEX
    position = (30, 60)
    font_scale = 1.2
    color = (0, 0, 255)   # Red (BGR)
    thickness = 2

    cv2.putText(img, text, position, font, font_scale, color, thickness, cv2.LINE_AA)
    return img


if __name__ == "__main__":
    if len(sys.argv) == 2:
        img = cv2.imread(sys.argv[1])
        if img is None:
            raise IOError(f"Cannot read image: {sys.argv[1]}")
    else:
        img = np.full((300, 700, 3), 255, dtype=np.uint8)

    img = put_text_on_image(img)

    cv2.imshow("Text on Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite("text_on_image.png", img)
    print("Saved: text_on_image.png")
