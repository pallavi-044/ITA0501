"""
30. Smile detection using OpenCV Haar Cascades.

Detects faces first, then searches for a smile within each face region.

Usage:
    python 30_smile_detection.py [image_path]
(If no image path is given, uses the webcam.)
"""

import cv2
import sys

FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
SMILE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_smile.xml"


def detect_smiles(frame, face_cascade, smile_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (fx, fy, fw, fh) in faces:
        cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (255, 0, 0), 2)
        roi_gray = gray[fy:fy + fh, fx:fx + fw]
        roi_color = frame[fy:fy + fh, fx:fx + fw]

        # Smile cascade needs a higher minNeighbors to avoid false positives
        smiles = smile_cascade.detectMultiScale(
            roi_gray, scaleFactor=1.7, minNeighbors=22
        )
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), 2)
            cv2.putText(frame, "Smiling", (fx, fy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return frame


def run_on_image(path, face_cascade, smile_cascade):
    img = cv2.imread(path)
    if img is None:
        raise IOError(f"Cannot read image: {path}")
    img = detect_smiles(img, face_cascade, smile_cascade)
    cv2.imshow("Smile Detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_on_webcam(face_cascade, smile_cascade):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam.")
    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = detect_smiles(frame, face_cascade, smile_cascade)
        cv2.imshow("Smile Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
    smile_cascade = cv2.CascadeClassifier(SMILE_CASCADE_PATH)
    if face_cascade.empty() or smile_cascade.empty():
        raise IOError("Failed to load Haar cascades.")

    if len(sys.argv) == 2:
        run_on_image(sys.argv[1], face_cascade, smile_cascade)
    else:
        run_on_webcam(face_cascade, smile_cascade)
