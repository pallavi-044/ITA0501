"""
36. Background subtraction based on color levels using OpenCV.

Two approaches are shown:
  (a) Color-range based subtraction: define a color range (e.g. a green
      screen background) with cv2.inRange and remove it.
  (b) Statistical background subtraction using cv2.createBackgroundSubtractorMOG2
      for video, which models the background over time.

Usage:
    python 36_background_subtraction.py video_path.mp4
"""

import cv2
import numpy as np
import sys


def color_range_background_removal(frame, lower_bgr, upper_bgr):
    """Removes pixels whose color falls within [lower_bgr, upper_bgr]
    (treated as background) and keeps everything else as foreground."""
    lower = np.array(lower_bgr, dtype=np.uint8)
    upper = np.array(upper_bgr, dtype=np.uint8)

    background_mask = cv2.inRange(frame, lower, upper)
    foreground_mask = cv2.bitwise_not(background_mask)

    foreground = cv2.bitwise_and(frame, frame, mask=foreground_mask)
    return foreground, background_mask


def mog2_background_subtraction(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    back_sub = cv2.createBackgroundSubtractorMOG2(
        history=500, varThreshold=16, detectShadows=True
    )

    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        fg_mask = back_sub.apply(frame)
        fg = cv2.bitwise_and(frame, frame, mask=fg_mask)

        cv2.imshow("Original", frame)
        cv2.imshow("Foreground Mask (MOG2)", fg_mask)
        cv2.imshow("Foreground Only", fg)

        if cv2.waitKey(30) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python 36_background_subtraction.py <video_path>")
        sys.exit(1)

    # Example: run the statistical MOG2 background subtractor on a video
    mog2_background_subtraction(sys.argv[1])

    # Example of color-range based removal on a single frame:
    # frame = cv2.imread("some_image.jpg")
    # Removing a greenish background (BGR lower/upper bounds)
    # fg, mask = color_range_background_removal(frame, (0, 100, 0), (100, 255, 100))
    # cv2.imshow("Foreground", fg); cv2.waitKey(0); cv2.destroyAllWindows()
