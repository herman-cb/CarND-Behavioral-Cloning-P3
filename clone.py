import csv
import cv2
import numpy as np

lines = []
with open("data/driving_log.csv") as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []
for line in lines:
    source_path=line[0]
    #filename = source_path.split("/")[-1]
    image = cv2.imread("data/"+source_path)
    #print("image shape = {}".format(image.shape))
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)

X_train = np.asarray(images)
y_train = np.asarray(measurements)

a = -0.5
b = 0.5
X_min = np.min(X_train)
X_max = np.max(X_train)
print("X_max = {}, X_min = {}".format(X_max, X_min))
import sys
print("Bytes used by X_train = {}".format(sys.getsizeof(X_train)))
X_train = X_train - X_min
X_train = X_train * (b-a)
#X_train = X_train.astype(np.float32)
print("Bytes used by X_train = {}".format(sys.getsizeof(X_train)))
print("Shape of X_train = {}".format(X_train.shape))
X_train_array = X_train.shape[0]*[None]
for i in np.arange(X_train.shape[0]):
    print("Processing image number {}".format(i))
    X_train_array[i] = X_train[i]/X_max-X_min
X_train = np.stack(X_train_array)
print("Shape of X_train = {}".format(X_train.shape))
X_train = X_train + a
#X_train = a + (b - a)*(X_train - X_min) / (X_max - X_min)

print("Shape of X_train = {}".format(X_train.shape))
print("Shape of y_train = {}".format(y_train.shape))


from keras.models import Sequential
from keras.layers import Flatten, Dense

model = Sequential()
model.add(Flatten(input_shape=(160, 320, 3)))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam', metrics = ['mean_squared_error'])
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=7)

model.save('model.h5')
