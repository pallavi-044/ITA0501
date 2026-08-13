import cv2
import numpy as np

img = cv2.imread("image.jpeg")

rows, cols = img.shape[:2]

pts1 = np.float32([[50,50],[300,50],[50,300],[300,300]])
pts2 = np.float32([[0,0],[300,0],[100,300],[300,300]])

matrix = cv2.getPerspectiveTransform(pts1, pts2)

result = cv2.warpPerspective(img, matrix, (cols, rows))

cv2.imshow("Perspective", result)
cv2.waitKey(0)
cv2.destroyAllWindows()