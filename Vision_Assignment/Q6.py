import cv2

img = cv2.imread("image.jpg", 0)  # grayscale

blur1 = cv2.GaussianBlur(img, (3, 3), 0)
blur2 = cv2.GaussianBlur(img, (9, 9), 0)

DoG = blur1 - blur2

cv2.imshow("Original", img)
cv2.imshow("DoG Filter", DoG)
cv2.waitKey(0)
cv2.destroyAllWindows()
