"""
27. Face detection using OpenCV Haar Cascades.

Works on an image or, if no image path is given, on the webcam feed.

Usage:
    python 27_face_detection.py [image_path]
"""

import cv2
import sys

FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


def detect_faces_in_frame(frame, face_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return frame, faces


def run_on_image(image_path, face_cascade):
    img = cv2.imread(image_path)
    if img is None:
        raise IOError(f"Cannot read image: {image_path}")
    img, faces = detect_faces_in_frame(img, face_cascade)
    print(f"Detected {len(faces)} face(s).")
    cv2.imshow("Face Detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_on_webcam(face_cascade):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam.")
    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame, _ = detect_faces_in_frame(frame, face_cascade)
        cv2.imshow("Face Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
    if face_cascade.empty():
        raise IOError("Failed to load Haar cascade for face detection.")

    if len(sys.argv) == 2:
        run_on_image(sys.argv[1], face_cascade)
    else:
        run_on_webcam(face_cascade)
