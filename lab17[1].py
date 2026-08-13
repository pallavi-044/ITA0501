import cv2

img = cv2.imread("image.jpeg")

cv2.putText(img, "SAVEETHA", (50,50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1, (255,255,255), 2)

cv2.imshow("Watermarked", img)
cv2.waitKey(0)
cv2.destroyAllWindows()