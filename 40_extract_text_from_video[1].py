"""
40. Extract text from videos using OpenCV + Tesseract OCR.

OpenCV itself does not perform OCR, so this script uses OpenCV to read and
preprocess frames, and pytesseract (a wrapper for Google's Tesseract OCR
engine) to extract text from those frames.

Requirements:
    pip install pytesseract opencv-python
    Also install the Tesseract OCR engine itself:
        - Ubuntu/Debian: sudo apt-get install tesseract-ocr
        - macOS:         brew install tesseract
        - Windows:       https://github.com/UB-Mannheim/tesseract/wiki

Usage:
    python 40_extract_text_from_video.py input_video.mp4 [frame_skip]

    frame_skip: process every Nth frame (default 30, i.e. ~1 frame/sec at 30fps)
"""

import cv2
import sys

try:
    import pytesseract
except ImportError:
    pytesseract = None


def preprocess_for_ocr(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def extract_text_from_video(video_path, frame_skip=30):
    if pytesseract is None:
        raise ImportError(
            "pytesseract is not installed. Run: pip install pytesseract "
            "and install the Tesseract OCR engine on your system."
        )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    frame_index = 0
    results = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % frame_skip == 0:
            processed = preprocess_for_ocr(frame)
            text = pytesseract.image_to_string(processed).strip()
            if text:
                timestamp_sec = frame_index / (cap.get(cv2.CAP_PROP_FPS) or 25)
                results.append((frame_index, timestamp_sec, text))
                print(f"[Frame {frame_index}, t={timestamp_sec:.2f}s] {text}")

        frame_index += 1

    cap.release()

    with open("extracted_text.txt", "w", encoding="utf-8") as f:
        for idx, ts, text in results:
            f.write(f"Frame {idx} (t={ts:.2f}s):\n{text}\n\n")

    print(f"\nDone. Extracted text from {len(results)} frame(s). "
          "Saved to extracted_text.txt")
    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python 40_extract_text_from_video.py <video_path> [frame_skip]")
        sys.exit(1)
    skip = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    extract_text_from_video(sys.argv[1], skip)
