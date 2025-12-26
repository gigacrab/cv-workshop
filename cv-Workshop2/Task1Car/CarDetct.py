from ultralytics import YOLO
import cv2
#0 for computer webcame
cap = cv2.VideoCapture("cars.mp4")
#set size of camera 3=width 4 =height we set it as this because later when we create mask we also create 1280 and 720
cap.set(3,1280)
cap.set(4,720)
#use canva to find out where the line should be
liney=416

model = YOLO('../YoloWeights/yolov8n.pt')
#use a mask to only allow detection for specific area
#mask is created using canva
mask=cv2.imread("MaskForYolo.png")
countcar =0
#use to store detections id for counting to prevent doublecounting
#the way we count is if car center point is in the area of the line then we count it
total_ids = []

while True:
    #success determines cap read true or false img is the numpy of pixels
    success, img=cap.read()
    if success==False:
        break  # This breaks the 'while True' loop
    regionfordetect = cv2.bitwise_and(img,mask)
    #draw a line linex and liney
    cv2.line(img, (256, liney), (545, liney), (0, 0, 255), 5)
    cv2.putText(img, f"Count:{countcar}", (72,38),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    #https://docs.ultralytics.com/modes/track/#features-at-a-glance
    #https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/trackers/bytetrack.yaml
    #pip install lap if it doesnt automatically update for you
    results = model.track(regionfordetect,tracker="custombytetrack.yaml",stream=True,persist=True)
    
    
    #results is an array results[0] means each image
    for i in results:
        #create an array for bounding boxes(for each object detected in a single image)
        boxes = i.boxes
        #for each box grab their x1 y1 x2 y2 corrdinates so we can draw boxes on them later
        for box in boxes:
            #get current box coordinates
            x1,y1,x2,y2 = box.xyxy[0]
            #x1y1 is top left corner x2y2 is bottom right corner
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
            print(x1,y1,x2,y2)
            #image,topleftcorner,bottomright corner,colour(rgb),thickness
            

            #get confidence text
            conf = box.conf[0]
            #combine int with string so for example now it becomes 80%
            conf_text = f'{int(conf * 100)}%'
            
            #box.cls gives u class name id but in float so change to int
            cls = int(box.cls[0])
            if box.id is not None:
                currentid = int(box.id[0])
            else:
                currentid = "" # Temporary placeholder
            
            #grab the current class name using model.names
            current_name = model.names[cls]
            if current_name == "car" or current_name == "bus" or current_name == "motorbike" and conf>0.2:
                cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),3)
                finalcombinedtext = f'{current_name} {currentid} {conf_text}%'
            #we use max to make sure text is inside
            #cv2.putText(img, current_name, ((x1 - 5, y1 - 5)),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                cv2.putText(img, finalcombinedtext, (max(0,x1), max(40,y1-20)),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                cx = int(x1+x2)//2
                cy = int(y1+y2)//2
                cv2.circle(img,(cx,cy),4,(255,0,0))
                if (liney - 20) < cy < (liney + 20):
                    if currentid not in total_ids:
                        total_ids.append(currentid)
                        countcar = len(total_ids)
                        cv2.line(img, (256, liney), (545, liney), (0, 255, 255), 5)

        
    #image show "Image" window name (img) conatins your img data from cap.read
    cv2.imshow("Image",img)

    #wait for key press to detect break or not
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()