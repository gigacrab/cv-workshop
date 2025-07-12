import cv2
import numpy as np

#img = np.zeros((512, 512, 3), np.uint8)

img = cv2.imread("assets/atla.jpg")
img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)

edge = cv2.Canny(img, 100, 200)

# Drawing a diagonal blue line with thickness 5
cv2.line(img, (0, 0), (511, 511), (255, 0, 0), 5)

# Drawing a rectangle coordinates of top left and bottom right corners
cv2.rectangle(img, (384, 0), (510, 128), (0, 255, 0), 3)

# Drawing a circle, centre and radius
cv2.circle(img, (447, 63), 63, (0, 0, 255), 1)

cv2.imshow("happy", img)
cv2.imshow("edge", edge)
while True:
    k = cv2.waitKey(1) & 0xFF
    if k == 27:  # Check for ESC key
        break
cv2.destroyAllWindows()
