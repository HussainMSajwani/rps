import cv2
import os
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import tensorflow as tf

imgs = []
labels = []


labs_dict = {
    "Rock": 0,
    "Paper": 1,
    "Scissors": 2
}

for move in ["Rock", "Paper", "Scissors"]:
    for img in os.listdir("001_data/" + move):
        imgs.append(cv2.imread(f"data/{move}/{img}"))
        labels.append(labs_dict[move])

imgs = np.array(imgs).astype('float32')
labels = tf.keras.utils.to_categorical(np.array(labels).reshape(-1, 1))

print("\n\n\n\n", imgs.shape)

model = Sequential([
    Conv2D(64, (3, 3), input_shape=(200, 200, 3), activation='relu'),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)), 
    Dropout(0.2),
    Conv2D(128, (3, 3), activation='relu'),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.2),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(3, activation='softmax')
])

model.compile(loss='categorical_crossentropy',
             optimizer='RMSprop',
             metrics=['accuracy', 'AUC'])

history = model.fit(imgs,
                    labels,
                    epochs=3,
                    verbose=1)

model.save("model1.h5")
