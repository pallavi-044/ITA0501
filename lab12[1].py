import cv2

img = cv2.imread("image.jpeg")

rotated = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

cv2.imshow("270 Degree", rotated)
cv2.waitKey(0)
cv2.destroyAllWindows()