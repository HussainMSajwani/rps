import cv2
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt


moves = ["Rock", "Paper", "Scissors"]

cap = cv2.VideoCapture(0)


lower = (0, 65, 0)
upper = (25, 173, 255)

ker = np.ones((5, 5))

def who_wins(my_move, pc_move):

    if pc_move == my_move:
        return "No one"
    elif pc_move == "Rock":
        if my_move == "Scissors":
            return "PC"
        else:
            return "Me"
    elif pc_move == "Scissors":
        if my_move == "Rock":
            return "Me"
        else:
            return "PC"
    elif pc_move == "Paper":
        if my_move == "Scissors":
            return "Me"
        else:
            return "PC"


def get_data(Id):
    imgs = []
    labels = []


    labs_dict = {
        "Rock": 0,
        "Paper": 1,
        "Scissors": 2
    }

    for move in ["Rock", "Paper", "Scissors"]:
        for img in os.listdir(f"{Id}_data/" + move):
            imgs.append(cv2.imread(f"{Id}_data/{move}/{img}"))
            labels.append(labs_dict[move])

    imgs = np.array(imgs).astype('float32')
    labels = np.array(labels)
    
    return imgs, labels

imgs_train, labels_train = get_data("003")

pca = PCA(20)
pca.fit(imgs_train.reshape(len(labels_train), 200*200*3))

X_train=pca.transform(imgs_train.reshape(len(labels_train), 200*200*3))

svm = SVC()
svm.fit(X_train, labels_train)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    frame = cv2.flip(frame, 1)
    frame = cv2.rectangle(frame, (400, 200), (600, 400), (255, 0, 0))

    cv2.imshow('skin', frame)

    square = frame[200:400, 400:600]
    hsv = cv2.cvtColor(square, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ker)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, ker)
    skin = cv2.bitwise_and(square, square, mask=mask)
    skin = np.array([cv2.flip(skin, 1)])
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        projected = pca.transform(skin.reshape(1, 200*200*3))
        prediction = svm.predict(projected)[0]

        pc_move = moves[np.random.randint(0, 3)]
        my_move = moves[prediction]

        print(f"my move: {my_move}\npc move: {pc_move}")
        print(f"{who_wins(my_move, pc_move)} wins \n\n\n\n")
    

cap.release()
cv2.destroyAllWindows()

