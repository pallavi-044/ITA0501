"""
38. Count the number of faces in a given input image using OpenCV.

Usage:
    python 38_count_faces.py input_image.jpg
"""

import cv2
import sys

FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


def count_faces(image_path):
    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
    if face_cascade.empty():
        raise IOError("Failed to load Haar cascade for face detection.")

    img = cv2.imread(image_path)
    if img is None:
        raise IOError(f"Cannot read image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    count = len(faces)
    cv2.putText(img, f"Faces: {count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    print(f"Number of faces detected: {count}")

    cv2.imshow("Face Count", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("face_count_result.png", img)
    return count


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python 38_count_faces.py <image_path>")
        sys.exit(1)
    count_faces(sys.argv[1])
