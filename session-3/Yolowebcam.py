from ultralytics import YOLO
import cv2
#0 for computer webcame
cap = cv2.VideoCapture(0)
#set size of camera 3=width 4 =height
cap.set(3,1280)
cap.set(4,720)

model = YOLO('../YoloWeights/yolov8n.pt')

while True:
    #success determines cap read true or false img is the numpy of pixels
    success, img=cap.read()
    results = model(img,stream=True)
    #results is an array results[0] means each image
    for i in results:
        #create an array for bounding boxes(for each object detected in a single image)
        boxes = i.boxes
        #for each box grab their x1 y1 x2 y2 corrdinates so we can draw boxes on them later
        for box in boxes:
            #get confidence value
            
            x1,y1,x2,y2 = box.xyxy[0]
            #x1y1 is top left corner x2y2 is bottom right corner
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
            print(x1,y1,x2,y2)
            #image,topleftcorner,bottomright corner,colour(rgb),thickness
            cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),3)

            #put confidence text
            conf = box.conf[0]
            #combine int with string so for example now it becomes 80%
            conf_text = f'{int(conf * 100)}%'
            #put text on img with conf_text
            cls = int(box.cls[0])
    
            
            #put object text
            current_name = model.names[cls]
            #we use max to make sure text is inside
            #cv2.putText(img, current_name, ((x1 - 5, y1 - 5)),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.putText(img, current_name, (max(0,x1), max(40,y1-20)),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    #image show "Image" window name (img) conatins your img data from cap.read
    cv2.imshow("Image",img)
    #wait for key press to detect break or not
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()