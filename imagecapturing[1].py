import cv2

cam = cv2.VideoCapture(0)

while True:
    ret, img = cam.read()
    cv2.imshow("Camera", img)

    key = cv2.waitKey(1)

    if key == ord('s'):
        cv2.imwrite("image.jpg", img)
        print("Image Captured!")
        break

    elif key == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()