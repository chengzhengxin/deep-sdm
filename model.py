import os
import sys
import random
import math
import time
import h5py
import cv2
import pickle
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from utils.util import *

class myDatasets(Dataset):
    def __init__(self, data_path):
        self.crop_size = config.crop_size
        self.channels  = 3 * len(config.sample_idx)
        self.landmark_size = config.landmark_num * 2

        self.h5f = h5py.File(data_path, 'r')
        self.feature  = self.h5f['feature']
        self.landmark = self.h5f['landmark']
        self.metadata = self.h5f['metadata']
        self.sample_len  = self.metadata.shape[0]
        self.zeros_mark  = np.zeros((self.landmark_size,), dtype=np.float32)

    def __len__(self):
        return self.sample_len
    
    def __getitem__(self, index):
        feature  = self.feature[index, :]
        landmark = self.landmark[index, :]
        metadata = self.metadata[index, :]

        feat_data = feature.reshape((self.crop_size, self.crop_size, self.channels))
        feat_data = feat_data.transpose(2, 0, 1)
        face_label = metadata[1]
        mark_label = self.zeros_mark if face_label == 0 else landmark[:self.landmark_size]

        return feat_data, mark_label, face_label


class WingLoss(nn.Module):
    def __init__(self, omega=10, epsilon=2):
        super(WingLoss, self).__init__()
        self.omega = omega
        self.epsilon = epsilon

    def forward(self, pred, target):
        y = target
        y_hat = pred
        delta_y = (y - y_hat).abs()
        delta_y1 = delta_y[delta_y < self.omega]
        delta_y2 = delta_y[delta_y >= self.omega]
        loss1 = self.omega * torch.log(1 + delta_y1 / self.epsilon)
        C = self.omega - self.omega * math.log(1 + self.omega / self.epsilon)
        loss2 = delta_y2 - C
        return (loss1.sum() + loss2.sum()) / (len(loss1) + len(loss2))

def weights_init(m):
    classname = m.__class__.__name__
    if 'Conv2d' in classname:
        nn.init.xavier_normal_(m.weight.data)
        nn.init.constant_(m.bias.data, 0.0)
    elif 'Linear' in classname:
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0.0)

class PNET(nn.Module):
    def __init__(self):
        super(PNET, self).__init__()

        groups = len(config.step1_idx)
        in_channels = groups * 3
        dims = [3, 16, 32, 64]

        self.m1 = nn.Sequential(
                        nn.BatchNorm2d(in_channels),
                        nn.Conv2d(in_channels, groups * dims[1], kernel_size=3, stride=1, padding=0, groups=groups, bias=True),
                        nn.BatchNorm2d(groups * dims[1]),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2)
                        )
        
        self.m2 =  nn.Sequential(
                        nn.Conv2d(groups * dims[1], groups * dims[2], kernel_size=3, stride=1, padding=1, groups=groups, bias=True),
                        nn.BatchNorm2d(groups * dims[2]),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=3, stride=2)
                        )

        self.m3 =  nn.Sequential(
                        nn.Conv2d(groups * dims[2], groups * dims[3], kernel_size=2, stride=1, padding=0, groups=groups, bias=True),
                        nn.ReLU()
                        )

        self.pd = nn.Sequential(
                        nn.Linear(groups * dims[3], 256),
                        nn.ReLU(),
                        nn.Linear(256, 2 * config.landmark_num)
                        )

    def forward(self, x):
        x = self.m1(x)
        x = self.m2(x)
        x = self.m3(x)
        x = x.view(x.size(0), -1)
        x = self.pd(x)
        return x

class RNET(nn.Module):
    def __init__(self, is_train=True):
        super(RNET, self).__init__()

        groups = len(config.step2_idx)
        in_channels = groups * 3
        dims = [3, 8, 16, 32]
        # dims = [3, 16, 32, 64]

        self.m1 = nn.Sequential(
                        nn.BatchNorm2d(in_channels),
                        nn.Conv2d(in_channels, groups * dims[1], kernel_size=3, stride=1, padding=0, dilation=1, groups=groups, bias=True),
                        nn.BatchNorm2d(groups * dims[1]),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2)
                        )
        
        self.m2 =  nn.Sequential(
                        nn.Conv2d(groups * dims[1], groups * dims[2], kernel_size=3, stride=1, padding=1, dilation=1, groups=groups, bias=True),
                        nn.BatchNorm2d(groups * dims[2]),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2)
                        )

        self.m3 =  nn.Sequential(
                        nn.Conv2d(groups * dims[2], groups * dims[3], kernel_size=2, stride=1, padding=0, dilation=1, groups=groups, bias=True),
                        nn.ReLU()
                        )

        self.m4 = nn.Sequential(
                        nn.Linear(groups * dims[3], 256),
                        nn.ReLU()
                        )
        
        self.m5 = nn.Linear(256, 2 * config.landmark_num)
        self.m6 = nn.Linear(256, 2)
        self.m7 = nn.LogSoftmax(dim=1) if is_train else nn.Softmax(dim=1)


    def forward(self, x):
        x = self.m1(x)
        x = self.m2(x)
        x = self.m3(x)

        x = x.view(x.size(0), -1)
        x = self.m4(x)

        y1 = self.m5(x)
        x  = self.m6(x)
        y2 = self.m7(x)

        return y1, y2


def get_pmodel(model_path=None):
    is_cuda = not config.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if is_cuda else 'cpu')
    model = PNET()
    if model_path is not None:
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint)
    else:
        model.apply(weights_init)
    model = model.to(device)
    return model, device

def get_rmodel(model_path=None, is_train=True):
    is_cuda = not config.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if is_cuda else 'cpu')
    model = RNET(is_train=is_train)
    if model_path is not None:
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint)
    else:
        model.apply(weights_init)
    model = model.to(device)
    return model, device

