from ultralytics import YOLO
import cv2
model = YOLO("yolov8n-pose.pt")
cap = cv2.VideoCapture("fall.mp4")
#set size of camera 3=width 4 =height
#https://www.youtube.com/watch?v=YKbBXWBJloY

#We check the person is falling or not by checking width>height
# we check if their nose is below their hip or below their knee
#https://docs.ultralytics.com/tasks/pose/
#https://docs.ultralytics.com/reference/engine/results/#ultralytics.engine.results.Keypoints
#refer to the above for each index for each keypoints

while True:

    success, img = cap.read()
    if success==False:
        break  # This breaks the 'while True' loop
    img=cv2.resize(img,(1280,720))
    results = model(img,stream=True)
    #vidoe finished

    
    
    
    #go through each image and get all the detected people from the images
    for r in results:
        #Use r.plot() to automatically get the skeletons provided by yolo
        annotated_frame = r.plot(labels=False, boxes=False)
        #if detected keypoints we grab them put in a list
        if r.keypoints is not None:
            #number of detected people,17 joints and xy coordinate
            kpt_list = r.keypoints.xy #(num_people, 17(joint), 2)
            kpconfidence = r.keypoints.conf#(num_people, 17(confidence each joint))
            num_people = len(kpt_list)
            boxes = r.boxes

            #Loop through each person (num_people, 17, 2)
            for personid in range(num_people):
                box = boxes[personid]#get this specific person bounding box
                Nosey, LeftKneeY, RightKneeY, LeftHipY, RightHipY = -1,-1,-1,-1,-1
                #Get coordinates for the text position
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2= int(x1), int(y1), int(x2),int(y2)
                width = x2 - x1
                height = y2 - y1
                #use if width > height more than 1.5 detect as falling
                ratioWH = width / height if height > 0 else 0
                is_horizontal = ratioWH > 1.5
        
                
                #get current person list of keypoints
                current_person_kpts = kpt_list[personid]
                #get current person list of keypoints confidence
                current_person_conf = kpconfidence[personid]
                print(f"Keypoints for one person:{personid}")
                #Add another loop to go through the 17 joints of this specific person
                # Loop through each joints(17, 2)
                for i in range(17):
                    #Get each joints x and y (2)
                    conf = current_person_conf[i]
                    point = current_person_kpts[i]#current person keypoint(one of 17)
                    x = int(point[0]) #keypoint x axis (one of 2)
                    y = int(point[1])
                    if conf>0.5:
                        
                        #Nose
                        if i == 0:
                            Nosey = int(point[1])
                        #Left Knee
                        if i == 13:
                            LeftKneeY = int(point[1])
                        #Right Knee
                        if i == 14:
                            RightKneeY = int(point[1])
                        #Left Hip
                        if i ==11:
                            LeftHipY = int(point[1])
                        #Right Hip
                        if i ==12:
                            RightHipY = int(point[1])

                #If all are detected we check for eveyrhting because if it didnt detect it will put garbage values
                #If added garbage values then will cause error in detection 
                if LeftKneeY > 0 and RightKneeY > 0 and RightHipY>0 and LeftHipY >0:
                    avg_Knee_y = (LeftKneeY + RightKneeY) / 2
                    avg_hip_y = (LeftHipY + RightHipY) / 2
                    

                    is_fallen = is_horizontal or (Nosey > avg_hip_y)
                    status_text = "Not falling"
                    box_color = (255, 0, 0) # Blue

                    if is_fallen:
                        status_text = "FALL DETECTED!"
                        box_color = (0, 0, 255) # Red

                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 3)
                    cv2.putText(annotated_frame, status_text, (x1, max(30, y1 - 10)),cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2)
                
                #if detected hip only then we only check with hip
                elif RightHipY>0 and LeftHipY >0:
                    avg_Knee_y = (LeftKneeY + RightKneeY) / 2
                    avg_hip_y = (LeftHipY + RightHipY) / 2
                    

                    is_fallen = is_horizontal or (Nosey > avg_hip_y)
                    status_text = "Not falling"
                    box_color = (255, 0, 0) #Blue

                    if is_fallen:
                        status_text = "FALL DETECTED!"
                        box_color = (0, 0, 255) #Red

                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 3)
                    cv2.putText(annotated_frame, status_text, (x1, max(30, y1 - 10)),cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2)

    cv2.imshow("FallDetection", annotated_frame)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()