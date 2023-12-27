import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation
import numpy as np
import cv2
import pickle
import glob

data_true = "../archive/yes/*.jpg"
data_false = "../archive/no/*.jpg"
x_train = []
y_train = (
    []
)  # As per code [] --first value is no(no brain tumor) and second is yes(brain tumor is there)

for i in glob.iglob(data_true):
    img = cv2.imread(i)
    img = cv2.resize(img, (128, 128))
    x_train.append(img)
    y_train.append([0, 1])

for i in glob.iglob(data_false):
    img = cv2.imread(i)
    img = cv2.resize(img, (128, 128))
    x_train.append(img)
    y_train.append([1, 0])

x_train = np.array(x_train)
y_train = np.array(y_train)

input_shape = x_train.shape[1:]


x_train = x_train.astype(np.float32)
y_train = y_train.astype(np.float32)

model = Sequential()

model.add(Conv2D(32, (5, 5), input_shape=input_shape))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (5, 5), input_shape=input_shape))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (5, 5)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(2))

model.add(Activation("softmax"))

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

model.fit(x_train, y_train, epochs=20, validation_split=0.1)

with open("model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)


print("Running Successfully")
