import os
import sys
import random
import math
import time
import numpy as np
import cv2
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from utils.util import *
from model import *


def train_pmodel():
    print('training start')
    train_dataset = myDatasets(os.path.join(config.data_path, 'ws/train_data_with_crop_step1.hdf5'))
    train_loader  = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    model, device = get_pmodel()

    if config.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    elif config.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum)
    else:
        optimizer = None
        pass
    scheduler = optim.lr_scheduler.StepLR(optimizer, config.lr_step, config.lr_gamma)

    loss_func = nn.MSELoss()
    # loss_func = nn.SmoothL1Loss()
    # loss_func = WingLoss()

    for epoch in range(config.epochs):
        model.train()
        for i, (datas, labels, _) in enumerate(train_loader):
            datas, labels = datas.to(device, dtype=torch.float), labels.to(device, dtype=torch.float)
            preds = model(datas)

            optimizer.zero_grad()
            loss = loss_func(preds, labels)
            loss.backward()
            optimizer.step()

            if i % config.log_interval == 0:
                date = time.strftime("%Y-%m-%d %H:%M:%S")
                print('{}\t epoch={:0>3}, batch={:0>4}, lr={:.5f}, loss={:.5f}'.format(date, epoch, i, optimizer.state_dict()['param_groups'][0]['lr'], loss.item()))
        scheduler.step()

        save_path = os.path.join(config.data_path, 'ws/models/model_step1_{:0>2}.pt'.format(epoch))
        torch.save(model.state_dict(),  save_path)

    print('training finished')
    exit()

def train_rmodel():
    print('training start')
    train_dataset = myDatasets(os.path.join(config.data_path, 'ws/train_data_with_crop_step2.hdf5'))
    train_loader  = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    model, device = get_rmodel()

    if config.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    elif config.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum)
    else:
        optimizer = None
        pass
    scheduler = optim.lr_scheduler.StepLR(optimizer, config.lr_step, config.lr_gamma)

    loss1 = nn.MSELoss(reduce=False)
    # loss1 = nn.SmoothL1Loss()
    # loss1 = WingLoss()
    loss2 = nn.NLLLoss()

    for epoch in range(config.epochs):
        model.train()
        for i, (datas, mark_labels, face_labels) in enumerate(train_loader):
            datas = datas.to(device, dtype=torch.float)
            mark_labels = mark_labels.to(device, dtype=torch.float)
            mark_weight = face_labels.to(device, dtype=torch.float)
            face_labels = face_labels.to(device, dtype=torch.long)

            mark_preds, face_preds = model(datas)

            optimizer.zero_grad()
            loss1_ = loss1(mark_preds, mark_labels).mean(dim=1) * mark_weight
            loss2_ = loss2(face_preds, face_labels)
            loss   = loss1_.mean() + config.class_weight * loss2_
            loss.backward()
            optimizer.step()

            if i % config.log_interval == 0:
                date = time.strftime("%Y-%m-%d %H:%M:%S")
                print('{}\t epoch={:0>2}, batch={:0>4}, lr={:.5f}, loss={:.5f}'.format(date, epoch, i, optimizer.state_dict()['param_groups'][0]['lr'], loss.item()))
        scheduler.step()

        save_path = os.path.join(config.data_path, 'ws/models/model_step2_{:0>2}.pt'.format(epoch))
        torch.save(model.state_dict(),  save_path)

    print('training finished')


if __name__ == '__main__':
    
    # train_pmodel()
    
    train_rmodel()

