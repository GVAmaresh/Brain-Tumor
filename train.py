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

from MRI import MRIDataset
from CNN import CNNDataset

def train_model():
    mri = MRIDataset()
    mri.normalize()
    mri.train_val_split()

    train_value = DataLoader(mri, batch_size=32, shuffle=True)
    test_value = DataLoader(mri, batch_size=32, shuffle=False)

    device = torch.device("cuda")
    model = CNNDataset().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01 )

    epochs_train_loss=[]
    epochs_test_loss=[]

    for epoch in range(1, 300):
        train_loss = []
        model.train()
        mri.mode = "train"
        for i in train_value:
            optimizer.zero_grad()
            data = i["image"].to(device)
            label = i["label"].to(device)
            y_hat = model(data)
            error = nn.BCELoss()
            loss = torch.sum(error(y_hat.squeeze(), label))
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        epochs_train_loss.append(np.mean(train_loss))

        test_loss = []
        model.eval()
        mri.mode="test"
        with torch.no_grad():
            for i in test_value:
                data = i["image"].to(device)
                label = i["label"].to(device)
                y_hat = model(data)
                error = nn.BCELoss()
                loss = torch.sum(error(y_hat.squeeze(), label))
                test_loss.append(loss.item())
        epochs_test_loss.append(np.mean(test_loss))

        if (epoch+1 )%10 == 0:
            print(f'Train Epoch: {epoch} Train Loss: {np.mean(train_loss)} Test Loss: {np.mean(test_loss)}')
    return model, epochs_train_loss, epochs_test_loss

model, epochs_train_loss, epochs_test_loss = train_model()


plt.figure(figsize=(16, 9))
plt.plot(epochs_train_loss, c='b', label="Train Loss")
plt.plot(epochs_test_loss, c='r', label="Validation Loss")
plt.legend()
plt.grid()
plt.xlabel("Epochs", fontsize=20)
plt.ylabel("Loss", fontsize=20)
# plt.show()