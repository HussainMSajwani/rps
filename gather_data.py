import numpy as np
import cv2
import time 
from os import mkdir, listdir
    

#SAMPLING
#id of sample. DO NOT CHANGE UNLESS YOU'RE HUSSAIN. 
user_id = "003"
NUM_OF_SAMPLES = 500
#FPS setup
prev_frame_time = 0
new_frame_time = 0
#HSV skin thresholding
lower = (0, 71, 0)
upper = (25, 173, 255)

cap = cv2.VideoCapture(0)
moves = ['Rock', 'Paper', 'Scissors']

#make data_rps directory and its subdirectories  

if not f"{user_id}_data" in listdir(): 
    mkdir(f"{user_id}_data")
    for move in moves:
        mkdir(f"{user_id}_data/{move}")

for i, move in enumerate(moves):
    n = 0   
    f = 0
    while(True):
        # pre-process frame. flip and add rectangle
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        frame = cv2.rectangle(frame, (400, 200), (600, 400), (255, 0, 0))
        #hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        #skin = cv2.inRange(hsv, lowerb = lower, upperb = upper)

        # FPS STUFF
        new_frame_time = time.time() 
        if f % 3 == 0:
            fps = np.floor(1/(new_frame_time-prev_frame_time)) 
        prev_frame_time = new_frame_time 
        cv2.putText(frame, f"FPS: {fps}", (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 3, cv2.LINE_AA) 

        cv2.imshow('skin', frame)
        f += 1
        print(f"Hold your hand in {move} formation. {move} pictures = {n}/{NUM_OF_SAMPLES}")
        if cv2.waitKey(1) & 0xFF == ord('c'):
            flipepd_rectangle = cv2.flip(frame[200:400, 400:600], 1)
            cv2.imwrite(f"{user_id}_data/{move}/{user_id}_{n+1}.jpeg", flipepd_rectangle)
            n += 1

        if n == NUM_OF_SAMPLES:
            break
    
    now = time.time()
    if not i==2:
        while(time.time() - now <= 3):
            print(f"Pause. Hold your hand in a {moves[i+1]} formation")

    # When everything done, release the capture
cap.release()
cv2.destroyAllWindows()