import csv
import cv2
import numpy as np
import pickle
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
import os
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Dropout, Activation, Merge
from keras.regularizers import l2, activity_l2

lines = []
with open("data/driving_log.csv") as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []
for line in lines:
    for i in range(3):
        source_pathc=line[i]
        filenamec = source_pathc.split("/")[-1]
        image = cv2.imread("data/IMG/"+filenamec)
        images.append(image)
    measurement = float(line[3])
    correction = 0.2
    measurements.append(measurement)
    measurements.append(measurement+correction)
    measurements.append(measurement-correction)

augmented_images = []
augmented_measurements = []

for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image, 1))
    augmented_measurements.append(measurement * -1.0)


X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

print("Shape of X_train = {}".format(X_train.shape))
print("Shape of y_train = {}".format(y_train.shape))

def get_model0():
    model = Sequential()
    model.add(Lambda( lambda x : x/255.0 - 0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((70, 25), (0,0))))
    model.add(Convolution2D(6,5,5, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.5))
    model.add(Convolution2D(16,5,5, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model 

def get_model1():
    model = Sequential()
    model.add(Lambda( lambda x : x/255.0 - 0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((70, 25), (0,0))))
    model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dropout(0.5))
    model.add(Dense(50))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model 

model = get_model1()
#if os.path.exists("model.h5"):
#    model = load_model("model.h5")
model_checkpoint = ModelCheckpoint("model.h5", monitor="val_loss", save_best_only=True, period=1, mode='min')
model.fit(X_train, y_train, batch_size=32, validation_split=0.2, shuffle=True, nb_epoch=10, callbacks=[model_checkpoint])

model.save('model_final.h5')
