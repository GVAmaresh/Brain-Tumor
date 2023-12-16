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


class CNNDataset(nn.Module):
    def __init__(self):
        super(CNNDataset,self).__init__()
        self.cnn_model = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5),
        nn.Tanh(),
        nn.AvgPool2d(kernel_size=2, stride=5),
        nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
        nn.Tanh(),
        nn.AvgPool2d(kernel_size=2, stride=5))

        self.fc_model = nn.Sequential(
        nn.Linear(in_features=256, out_features=120),
        nn.Tanh(),
        nn.Linear(in_features=120, out_features=84),
        nn.Tanh(),
        nn.Linear(in_features=84, out_features=1))

    def forward(self, x):
        x = self.cnn_model(x)
        x = x.view(x.size(0), -1)
        x = self.fc_model(x)
        x = F.sigmoid(x)

        return x
    def predicted_value(self, path):
        img = cv2.imread(path)
        img = cv2.resize(img, (128, 128))
        if img is None:
            print(f"Error: Unable to load image at path: {path}")
        else:
            print("Image loaded successfully.")
        b, g, r = cv2.split(img)
        img = cv2.merge([r, g, b])
        img = img.reshape((img.shape[2], img.shape[0], img.shape[1]))
        img = np.array([img], dtype=np.float32)
        img = img/255.0
        img = torch.from_numpy(img).to(device,dtype=torch.float32)
        # img = img.permute(0, 3, 1, 2)
        # img = img.unsqueeze(0)
        with torch.no_grad():
          self.eval()
          print(self(img))
          prediction = self(img).squeeze().cpu().numpy()
        return prediction