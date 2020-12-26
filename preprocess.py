import cv2
import numpy as np

lower = (0, 71, 0)
upper = (25, 173, 255)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame = cv2.rectangle(frame, (400, 200), (600, 400), (255, 0, 0))

    square = frame[200:400, 400:600]

    hsv = cv2.cvtColor(square, cv2.COLOR_BGR2HSV)
    skin = cv2.inRange(hsv, lower, upper)
    skin = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, np.ones((5, 5)))
    skin = cv2.bitwise_and(square, square, mask=skin)
    cv2.imshow('skin', skin)
    #cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
