import cv2

img = cv2.imread("image.jpg")           # Read image (BGR format)
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
