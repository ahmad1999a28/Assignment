import cv2

img = cv2.imread("image.jpg")

choice = input("Choose transformation: darken / lighten / invert: ").lower()

if choice == "darken":
    result = img - 50          # subtract brightness
elif choice == "lighten":
    result = img + 50          # add brightness
elif choice == "invert":
    result = 255 - img         # invert colors
else:
    print("Invalid choice.")
    exit()

cv2.imshow("Original", img)
cv2.imshow("Result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
