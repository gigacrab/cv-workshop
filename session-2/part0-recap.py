import cv2

# initiate video capture at index 0
cap = cv2.VideoCapture(0)

while True:
    # ret - boolean indicating if frame was read successfully
    # frame - NumPy array representing image
    ret, frame = cap.read()

    # simple image processing, convert to a gray image from BGR
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # show frame image with window named "frame"
    cv2.imshow("frame", gray)

    # wait for ESC key for 1ms
    if cv2.waitKey(1) == 27:
        break

# releases video capture
cap.release()

# close GUI windows
cv2.destroyAllWindows()
