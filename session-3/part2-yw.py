import cv2
import numpy as np
from ultralytics import YOLOWorld

model = YOLOWorld("yolov8x-worldv2.pt")

classes = ["tennis ball", "tennis racket", "person", "tennis net"]
model.set_classes(classes)

cap = cv2.VideoCapture("wii-sports.mp4")

# To use device camera
# cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Failed to open video")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_AREA)

    # Conversion to RGB not needed, model's internal preprocessing handles that
    results = model.predict(source=frame, verbose=False, conf=0.25, save_dir=None)

    # Convert to BGR numpy array    
    annotated_frame = results[0].plot()

    # Note that this uses the BGR format
    cv2.imshow("Camera", annotated_frame)

    # ESC key
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()