from tensorflow.keras.models import load_model
from tensorflow import device
import cv2
import numpy as np

with device('/CPU:0'):
    model = load_model("model.h5").predict
cap = cv2.VideoCapture(0)
moves = ["Rock", "Paper", "Scissors"]
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    frame = cv2.flip(frame, 1)
    frame = cv2.rectangle(frame, (400, 200), (600, 400), (255, 0, 0))

    cv2.imshow('skin', frame)
    square = [cv2.flip(frame[200:400, 400:600], 1)]
    #print(f"{move} n = {n}/{NUM_OF_SAMPLES}")
    with device('/CPU:0'):
        prediction = np.argmax(model(np.array(square)))[0]
    print(moves[prediction])
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()