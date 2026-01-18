import cv2
import numpy as np
from keras.models import load_model

cap = cv2.VideoCapture(0)

model = load_model("tm-model/keras_model.h5", compile=False)

class_names = open("tm-model/labels.txt", "r").readlines()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    x = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    x = cv2.resize(x, (224, 224), interpolation=cv2.INTER_AREA).astype(np.float32)
    x = np.expand_dims(x, axis=0)

    # Normalizing -1 to 1 / preprocess_input(x)
    x = (x / 127.5) - 1
    preds = model.predict(x)

    # Log results
    print(preds)

    # Returns indices of the maximum value
    index = np.argmax(preds)
    class_name = class_names[index]

    # Print output
    confidence_score = preds[0][index]
    text = f"{class_name[2:-1]}: {np.round(confidence_score*100)}"

    # img, text, bottom left, font, scaling, color, thickness
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    cv2.imshow("Camera", frame)

    # ESC key
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()