import cv2

img = cv2.imread("image.jpeg")

rotated = cv2.rotate(img, cv2.ROTATE_180)

cv2.imshow("180 Degree", rotated)
cv2.waitKey(0)
cv2.destroyAllWindows()