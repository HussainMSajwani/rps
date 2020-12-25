import cv2
import numpy as np
import time

#Open Camera object
cap = cv2.VideoCapture(0)

#Decrease frame size

def nothing(x):
    pass

# Function to find angle between two vectors
def Angle(v1,v2):
 dot = np.dot(v1,v2)
 x_modulus = np.sqrt((v1*v1).sum())
 y_modulus = np.sqrt((v2*v2).sum())
 cos_angle = dot / x_modulus / y_modulus
 angle = np.degrees(np.arccos(cos_angle))
 return angle

# Function to find distance between two points in a list of lists
def FindDistance(A,B): 
 return np.sqrt(np.power((A[0][0]-B[0][0]),2) + np.power((A[0][1]-B[0][1]),2)) 
 

# Creating a window for HSV track bars
cv2.namedWindow('HSV_TrackBar')

# Starting with 100's to prevent error while masking
h,s,v = 100,100,100

# Creating track bar
cv2.createTrackbar('h', 'HSV_TrackBar',0,179,nothing)
cv2.createTrackbar('s', 'HSV_TrackBar',0,255,nothing)
cv2.createTrackbar('v', 'HSV_TrackBar',0,255,nothing)

while(1):

    #Measure execution time 
    start_time = time.time()
    
    #Capture frames from the camera
    ret, frame = cap.read()
    cv2.imshow("da", frame)
    #close the output video by pressing 'ESC'
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break


cap.release()
cv2.destroyAllWindows()