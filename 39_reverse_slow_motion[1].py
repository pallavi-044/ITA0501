"""
39. Play the given video in reverse mode with slow motion using OpenCV.

Reads all frames, then plays them back in reverse order, with each frame
held on screen longer than its original duration to create a slow-motion
effect (and optionally repeats frames for an even slower playback).

Usage:
    python 39_reverse_slow_motion.py input_video.mp4 [slow_factor]

    slow_factor: how many times slower than normal speed (default 2 = half speed)
"""

import cv2
import sys


def play_reverse_slow_motion(video_path, slow_factor=2):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_delay_ms = int(1000 / fps) * slow_factor

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        raise ValueError("No frames read from input video.")

    print(f"Playing {len(frames)} frames in reverse at {slow_factor}x slower. "
          "Press 'q' to quit.")

    for frame in reversed(frames):
        cv2.imshow("Reverse Slow Motion", frame)
        if cv2.waitKey(frame_delay_ms) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python 39_reverse_slow_motion.py <video_path> [slow_factor]")
        sys.exit(1)
    factor = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    play_reverse_slow_motion(sys.argv[1], factor)
