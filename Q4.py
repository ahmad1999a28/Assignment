import cv2
import numpy as np

img = cv2.imread("image.jpg", 0)   # Read in grayscale

low = 120      # change values to test
high = 180

result = img.copy()
result[(img >= low) & (img <= high)] = 255

cv2.imshow("Original", img)
cv2.imshow("Selective Thresholding", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
