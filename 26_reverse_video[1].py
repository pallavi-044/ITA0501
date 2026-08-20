"""
26. Reverse the frames of a video and create a reversed video using OpenCV.

Usage:
    python 26_reverse_video.py input_video.mp4 output_reversed.mp4
"""

import cv2
import sys


def reverse_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Read all frames into memory
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        raise ValueError("No frames read from input video.")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Write frames in reverse order
    for frame in reversed(frames):
        out.write(frame)

    out.release()
    print(f"Reversed video saved to: {output_path} ({len(frames)} frames)")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python 26_reverse_video.py <input_video> <output_video>")
        sys.exit(1)
    reverse_video(sys.argv[1], sys.argv[2])
