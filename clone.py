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
