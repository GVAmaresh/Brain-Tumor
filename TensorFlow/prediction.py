import cv2
import numpy as np
import pickle

model = pickle.load(open("./model.pkl", "rb"), encoding="latin1")

image_path = "../archive/brain_tumor_dataset/no/15 no.jpg"
img = cv2.imread(image_path)
img = cv2.resize(img, (128, 128))
img = np.expand_dims(img, axis=0)
img = img.astype(np.float32)

prediction = model.predict(img)
prediction = prediction.squeeze()
print(prediction)

if prediction[0] > prediction[1]:
    print("There is no Brain Tumor")
else:
    print("There is a Brain Tumor")
