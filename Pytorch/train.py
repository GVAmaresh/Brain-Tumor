from typing import Sequence
import torch
from torch.utils.data import DataLoader, TensorDataset
import glob
import cv2
import numpy as np

data_true = "../archive/yes/*.jpg"
data_false = "../archive/no/*.jpg"
x_train = []
y_train = []

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

x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)

dataset = TensorDataset(x_train_tensor, y_train_tensor)

batch_size = 64
shuffle = True
num_workers = 4
dataloader = DataLoader(
    dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
)


print("Running Successfully...")
