from ultralytics import YOLO
import cv2

cap = cv2.VideoCapture('cars.mp4')

model = YOLO("yolov8n.pt")
ListId = []
Liney = 495
linexstart = 148
linexend = 497
countcar = 0
mask=cv2.imread("MaskForYolo.png")
while True:
    success,frame = cap.read()
    framefordetect = cv2.bitwise_and(frame,mask)
    if success is False:
        break
    results = model.track(frame,persist=True)

    result = results[0]
    cv2.putText(frame,f'Count:{countcar}',(72,38),0,1,(255,0,0),2)
    cv2.line(frame,(148,495),(497,495),(0,255,0),3)

    for box in result.boxes:

        classint = int(box.cls[0])
        classname = result.names[classint]
        confcurrent = box.conf[0]
        print(result.boxes)

        if classname == 'car' and confcurrent>0.5:
            if box.id is not None:
                currentid = box.id[0]
            else:
                currentid=''
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
            cx = int((x1+x2)/2)
            cy = int((y1+y2)/2)
            cv2.circle(frame,(cx,cy),4,(255,0,0),-1)
            cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)

            cv2.putText(frame,f'{classname}{confcurrent:.1f}Id:{currentid}',(x1,max(10,y1-10)),0,1,(255,0,0),2)
            if linexstart<cx<linexend and Liney-20<cy<Liney+20 and currentid not in ListId:
                countcar+=1
                ListId.append(currentid)
                cv2.line(frame,(linexstart,Liney),(linexend,Liney),(0,0,255),5)

            
    cv2.imshow("img",frame)
    if cv2.waitKey(0)==ord('q'):
        break



cap.release()
cv2.destroyAllWindows()


