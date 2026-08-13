import cv2

img = cv2.imread("image.jpeg")

roi = img[50:200, 50:200]

img[250:400, 250:400] = roi

cv2.imshow("ROI", img)
cv2.waitKey(0)
cv2.destroyAllWindows()