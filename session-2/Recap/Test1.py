
#pip install opencv-python
#pip install ultralytics
#python -m venv myvenv
#problem with mingw old versio need new version
#go windowlib download 
#python version 3.12
from ultralytics import YOLO
import cv2 
model = YOLO('../YoloWeights/yolov8n.pt')
# Load the model


# Run inference - set show=False so we can control it manually
results = model("Images/2.jpeg", show=False)

# results is a list, so we take the first element
res = results[0]

# Plot the results onto the image (this returns a numpy array)
annotated_frame = res.plot()

# Display using standard OpenCV
cv2.imshow("YOLOv8 Detection", annotated_frame)

# This will now work as expected and wait for a key press
cv2.waitKey(0)
cv2.destroyAllWindows()

#Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
#.\.venv\Scripts\activate