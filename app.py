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

import torch
print(torch.cuda.is_available())

from train import train_model
model, epochs_train_loss, epochs_test_loss = train_model()

def have_tumor(value):
  return "Having Brain Tumor" if value>0.5 else "No Brain Tumor"

img_path = "./archive/brain_tumor_dataset/no/15 no.jpg"
predict_value = model.predicted_value(img_path)
print(predict_value, have_tumor(predict_value))



