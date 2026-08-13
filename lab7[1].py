import cv2

cap = cv2.VideoCapture("video.mp4.mp4")   # Use 0 for webcam

mode = "Normal"

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    cv2.putText(frame, mode, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)

    cv2.imshow("Video Processing", frame)

    # Set delay based on mode
    if mode == "Normal":
        delay = 30
    elif mode == "Slow":
        delay = 100
    elif mode == "Fast":
        delay = 5

    key = cv2.waitKey(delay) & 0xFF

    if key == ord('n'):
        mode = "Normal"
    elif key == ord('s'):
        mode = "Slow"
    elif key == ord('f'):
        mode = "Fast"
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()