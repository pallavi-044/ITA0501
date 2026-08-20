"""
29. Eye detection using OpenCV Haar Cascades.

Detects faces first, then searches for eyes within each face region
(this is more reliable than searching the whole frame).

Usage:
    python 29_eye_detection.py [image_path]
(If no image path is given, uses the webcam.)
"""

import cv2
import sys

FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
EYE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_eye.xml"


def detect_eyes(frame, face_cascade, eye_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    total_eyes = 0
    for (fx, fy, fw, fh) in faces:
        cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (255, 0, 0), 2)
        roi_gray = gray[fy:fy + fh, fx:fx + fw]
        roi_color = frame[fy:fy + fh, fx:fx + fw]

        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=8)
        total_eyes += len(eyes)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 255), 2)

    return frame, total_eyes


def run_on_image(path, face_cascade, eye_cascade):
    img = cv2.imread(path)
    if img is None:
        raise IOError(f"Cannot read image: {path}")
    img, eye_count = detect_eyes(img, face_cascade, eye_cascade)
    print(f"Detected {eye_count} eye(s).")
    cv2.imshow("Eye Detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_on_webcam(face_cascade, eye_cascade):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam.")
    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame, _ = detect_eyes(frame, face_cascade, eye_cascade)
        cv2.imshow("Eye Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
    eye_cascade = cv2.CascadeClassifier(EYE_CASCADE_PATH)
    if face_cascade.empty() or eye_cascade.empty():
        raise IOError("Failed to load Haar cascades.")

    if len(sys.argv) == 2:
        run_on_image(sys.argv[1], face_cascade, eye_cascade)
    else:
        run_on_webcam(face_cascade, eye_cascade)
