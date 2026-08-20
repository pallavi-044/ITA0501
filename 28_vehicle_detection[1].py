"""
28. Vehicle detection in each frame of a video using OpenCV.

Uses a Haar Cascade trained for cars (cars.xml). OpenCV does not ship this
cascade by default, so download it once, e.g.:

    curl -o cars.xml https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades_cuda/../../../3rdparty/... 

A commonly used community cascade is available here (search "cars.xml
haarcascade opencv" if the link below changes):
    https://github.com/andrewssobral/vehicle_detection_haarcascades

Place the downloaded cars.xml in the same folder as this script (or pass
its path as the third argument).

Usage:
    python 28_vehicle_detection.py input_video.mp4 output_video.mp4 [cars.xml]
"""

import cv2
import sys


def detect_vehicles(input_path, output_path, cascade_path="cars.xml"):
    car_cascade = cv2.CascadeClassifier(cascade_path)
    if car_cascade.empty():
        raise IOError(
            f"Failed to load car cascade from '{cascade_path}'. "
            "Download a Haar cascade trained on vehicles and place it here."
        )

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cars = car_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)

        for (x, y, w, h) in cars:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, "Vehicle", (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        out.write(frame)

    cap.release()
    out.release()
    print(f"Processed {frame_count} frames. Output saved to: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python 28_vehicle_detection.py <input_video> <output_video> [cars.xml]")
        sys.exit(1)
    cascade = sys.argv[3] if len(sys.argv) > 3 else "cars.xml"
    detect_vehicles(sys.argv[1], sys.argv[2], cascade)
