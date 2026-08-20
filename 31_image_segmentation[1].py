"""
31. Image segmentation based on threshold values using OpenCV.

Demonstrates simple binary thresholding, Otsu's thresholding, and
adaptive thresholding for segmenting an image into foreground/background.

Usage:
    python 31_image_segmentation.py input_image.jpg [threshold_value]
"""

import cv2
import sys


def segment_image(image_path, thresh_value=127):
    img = cv2.imread(image_path)
    if img is None:
        raise IOError(f"Cannot read image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 1. Simple binary threshold
    _, binary = cv2.threshold(blurred, thresh_value, 255, cv2.THRESH_BINARY)

    # 2. Otsu's threshold (automatically finds an optimal threshold)
    otsu_val, otsu = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    print(f"Otsu's optimal threshold: {otsu_val}")

    # 3. Adaptive threshold (useful for uneven lighting)
    adaptive = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

    cv2.imshow("Original", img)
    cv2.imshow(f"Binary Threshold ({thresh_value})", binary)
    cv2.imshow("Otsu Threshold", otsu)
    cv2.imshow("Adaptive Threshold", adaptive)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite("segmented_binary.png", binary)
    cv2.imwrite("segmented_otsu.png", otsu)
    cv2.imwrite("segmented_adaptive.png", adaptive)
    print("Saved: segmented_binary.png, segmented_otsu.png, segmented_adaptive.png")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python 31_image_segmentation.py <image_path> [threshold_value]")
        sys.exit(1)
    thresh = int(sys.argv[2]) if len(sys.argv) > 2 else 127
    segment_image(sys.argv[1], thresh)
