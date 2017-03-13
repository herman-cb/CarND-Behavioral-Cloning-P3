import csv
import cv2
import numpy as np
import pickle
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
import os
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Dropout, Activation


lines = []
with open("data/driving_log.csv") as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []
for line in lines:
    source_path=line[0]
    filename = source_path.split("/")[-1]
    image = cv2.imread("data/IMG/"+filename)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)
    images.append(np.fliplr(image))
    measurements.append(0-measurement)

X_train = np.array(images)
y_train = np.array(measurements)

print("Measurement range = {}-{}".format(np.min(measurements), np.max(measurements)))

print("Shape of X_train = {}".format(X_train.shape))
print("Shape of y_train = {}".format(y_train.shape))

# f = open("data.pkl", "wb")
# pickle.dump({"x": X_train, "y": y_train}, f)




# data = pickle.load(open("data.pkl", "rb"))

# X_train = data["x"]
# y_train = data["y"]

# x_max = np.max(X_train)
# x_min = np.min(X_train)
# X_train = -0.5 + (X_train-x_min)/(x_max-x_min)

print("Shape of X_train = {}".format(X_train.shape))
print("Shape of y_train = {}".format(y_train.shape))

model = Sequential()
model.add(Lambda( lambda x : x/255 - 0.5, input_shape=(160, 320, 3)))
model.add(Convolution2D(16, 3, 3, border_mode='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Activation('sigmoid'))
model.add(Dropout(0.5))
model.add(Convolution2D(8, 3, 3, border_mode='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Activation('sigmoid'))
model.add(Dropout(0.5))
model.add(Convolution2D(4, 3, 3, border_mode='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Activation('sigmoid'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100))
model.add(Activation('sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(20))
model.add(Activation('sigmoid'))
model.add(Dense(1))
model.add(Activation('tanh'))
model.compile(loss='mse', optimizer='adam')
#if os.path.exists("model.h5"):
#    model = load_model("model.h5")
model_checkpoint = ModelCheckpoint("model.h5", monitor="val_loss", save_best_only=True, period=1)
model.fit(X_train, y_train, batch_size=64, validation_split=0.2, shuffle=True, nb_epoch=35, callbacks=[model_checkpoint])

model.save('model.h5')
