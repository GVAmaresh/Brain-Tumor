import glob
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import glob
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
import random
import cv2
import torch.nn as nn
import torch.nn.functional as F
import sys
device = torch.device("cuda")

class Dataset(object):
    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __add__(self, other):
        return ConcatDataset([self, other])

class MRIDataset(Dataset):
    def __init__(self):
        self.mode = 'train'
        tumor = []
        healthy = []
        self.x_train, self.x_test, self.y_train, self.y_test = None, None, None, None
        for i in glob.iglob("./archive/brain_tumor_dataset/no/*.jpg"):
            img = cv2.imread(i)
            img = cv2.resize(img, (128, 128))
            b, g, r = cv2.split(img)
            img = cv2.merge([r, g, b])
            img = img.reshape((img.shape[2], img.shape[0], img.shape[1]))
            tumor.append(img)

        for i in glob.iglob("./archive/brain_tumor_dataset/yes/*.jpg"):
            img = cv2.imread(i)
            img = cv2.resize(img, (128, 128))
            b, g, r = cv2.split(img)
            img = cv2.merge([r, g, b])
            img = img.reshape((img.shape[2], img.shape[0], img.shape[1]))
            healthy.append(img)

        tumor = np.array(tumor, dtype=np.float32)
        healthy = np.array(healthy, dtype=np.float32)

        tumor_labels = np.ones(tumor.shape[0], dtype=np.float32)
        healthy_labels = np.ones(healthy.shape[0], dtype=np.float32)

        self.images = np.concatenate((tumor, healthy), axis=0)
        self.labels = np.concatenate((tumor_labels, healthy_labels), axis=0)

    def train_val_split(self):
      if self.images.shape[0] > 0:
          self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
              self.images, self.labels, test_size=0.2, random_state=42
          )
      else:
          raise ValueError("Dataset is empty. Ensure images are loaded correctly.")

    def __len__(self):
        if self.mode == "train":
            return self.x_train.shape[0]
        if self.mode == "test":
            return self.x_test.shape[0]

    def __getitem__(self, index):
        if self.mode == "train":
            return {"image": self.x_train[index], "label": self.y_train[index]}
        if self.mode == "test":
            return {"image": self.x_test[index], "label": self.y_test[index]}

    def normalize(self):
        self.images = self.images / 255.0