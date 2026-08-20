"""
37. Foreground subtraction based on color levels using OpenCV.

Extracts a foreground object by isolating pixels within a specific color
range (e.g. skin tone, a colored object) using cv2.inRange, then applies
the resulting mask to keep only the foreground and discard everything else.

Usage:
    python 37_foreground_subtraction.py image_or_video_path
"""

import cv2
import numpy as np
import sys
import os


def extract_foreground_by_color(frame, lower_hsv, upper_hsv):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(lower_hsv), np.array(upper_hsv))

    # Clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    foreground = cv2.bitwise_and(frame, frame, mask=mask)
    return foreground, mask


def process_image(path, lower_hsv, upper_hsv):
    img = cv2.imread(path)
    if img is None:
        raise IOError(f"Cannot read image: {path}")
    fg, mask = extract_foreground_by_color(img, lower_hsv, upper_hsv)

    cv2.imshow("Original", img)
    cv2.imshow("Mask", mask)
    cv2.imshow("Foreground", fg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("foreground_extracted.png", fg)
    print("Saved: foreground_extracted.png")


def process_video(path, lower_hsv, upper_hsv):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")

    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        fg, mask = extract_foreground_by_color(frame, lower_hsv, upper_hsv)
        cv2.imshow("Original", frame)
        cv2.imshow("Foreground", fg)
        if cv2.waitKey(30) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python 37_foreground_subtraction.py <image_or_video_path>")
        sys.exit(1)

    path = sys.argv[1]
    # Example HSV range: adjust to match the color of the object you want
    # to extract as foreground (this example targets a green object).
    lower_hsv = (35, 40, 40)
    upper_hsv = (85, 255, 255)

    ext = os.path.splitext(path)[1].lower()
    if ext in (".mp4", ".avi", ".mov", ".mkv"):
        process_video(path, lower_hsv, upper_hsv)
    else:
        process_image(path, lower_hsv, upper_hsv)
