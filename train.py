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

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def test_rmodel():
    pmodel, device = get_pmodel(os.path.join(config.data_path, 'ws/models/model_step1_35.pt'))
    pmodel.eval()

    W = np.load(os.path.join(config.data_path, 'ws/models/W1.npy'))
    landmark_pca_feat = np.load(os.path.join(config.data_path, 'ws/models/landmark_pca_feat.npy'))

    m = config.step1_m if config.active_step == 1 else config.step2_m
    scale = config.step1_scale if config.active_step == 1 else config.step2_scale
    resize = config.step1_resize if config.active_step == 1 else config.step2_resize
    crop_size = config.step1_crop_size if config.active_step == 1 else config.step2_crop_size
    sample_idx = config.step1_idx if config.active_step == 1 else config.step2_idx

    all_samples, all_landmarks, all_mirrors = load_all_ibug_datas()

    sample_num = all_landmarks.shape[0]
    for i in range(0, sample_num):
        target_landmark = all_landmarks[i, :]
        M = get_random_m(target_landmark, m[0], m[1], m[2])
        apply_similar_transform(target_landmark, M)
        proposal_landmark = reconstruct_landmark_from_pca(target_landmark, landmark_pca_feat, 1)
        image = cv2.imread(os.path.join(config.data_path, all_samples[i]))
        if all_mirrors[i]:
            image = cv2.flip(image, 1)

        start_time = time.time()
        feats = crop_feature(image, proposal_landmark, scale, resize, sample_idx, crop_size)
        feats = np.concatenate(feats, axis=2)
        feats = feats.transpose(2, 0, 1)[np.newaxis, ...]

        pmodel.eval()
        with torch.no_grad():
            pred = pmodel(torch.from_numpy(feats).to(device, dtype=torch.float))
        pred = pred.numpy()[0]
        pred_landmark = proposal_landmark
        landmarky = pred_landmark[1::2]
        miny = np.min(landmarky)
        maxy = np.max(landmarky)
        pred_landmark = pred_landmark + pred * (maxy - miny) / 100.
        print('time: {:.3f}ms'.format(time.time() - start_time))

        for i in range(config.landmark_num):
            cv2.circle(image, (int(proposal_landmark[2*i]), int(proposal_landmark[2*i + 1])), 2, (0,0,255), -1)
            cv2.circle(image, (int(pred_landmark[2*i]), int(pred_landmark[2*i + 1])), 2, (255,0,0), -1)
        cv2.imshow('image', image)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
    exit()

def test_rmodel():
    model, device = get_rmodel(os.path.join(config.data_path, 'ws/models/model_step2_03.pt'))
    model.eval()

    W = np.load(os.path.join(config.data_path, 'ws/models/W1.npy'))
    landmark_pca_feat = np.load(os.path.join(config.data_path, 'ws/models/landmark_pca_feat.npy'))

    m = config.step1_m if config.active_step == 1 else config.step2_m
    scale = config.step1_scale if config.active_step == 1 else config.step2_scale
    resize = config.step1_resize if config.active_step == 1 else config.step2_resize
    crop_size = config.step1_crop_size if config.active_step == 1 else config.step2_crop_size
    sample_idx = config.step1_idx if config.active_step == 1 else config.step2_idx

    all_samples, all_landmarks, all_mirrors = load_all_ibug_datas()

    sample_num = all_landmarks.shape[0]
    for i in range(300, sample_num, 2):
        target_landmark = all_landmarks[i, :]
        M = get_random_m(target_landmark, m[0], m[1], m[2])
        apply_similar_transform(target_landmark, M)
        proposal_landmark = reconstruct_landmark_from_pca(target_landmark, landmark_pca_feat, 12)
        image = cv2.imread(os.path.join(config.data_path, all_samples[i]))
        if all_mirrors[i]:
            image = cv2.flip(image, 1)

        start_time = time.time()
        feats = crop_feature(image, proposal_landmark, scale, resize, sample_idx, crop_size)
        feats = np.concatenate(feats, axis=2)
        feats = feats.transpose(2, 0, 1)[np.newaxis, ...]

        model.eval()
        with torch.no_grad():
            pred, isface = model(torch.from_numpy(feats).to(device, dtype=torch.float))
            pred = pred.numpy()[0]
            isface = isface.numpy()[0]
        print(isface)
        landmarky = proposal_landmark[1::2]
        miny = np.min(landmarky)
        maxy = np.max(landmarky)
        pred_landmark = proposal_landmark + pred * (maxy - miny) / 100.
        print('time: {:.3f}ms'.format(time.time() - start_time))

        for i in range(config.landmark_num):
            cv2.circle(image, (int(proposal_landmark[2*i]), int(proposal_landmark[2*i + 1])), 2, (0,0,255), -1)
            cv2.circle(image, (int(pred_landmark[2*i]), int(pred_landmark[2*i + 1])), 2, (255,0,0), -1)
        cv2.imshow('image', image)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
    exit()


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
                print('epoch={:0>3}, batch={:0>4}, lr={:.5f}, loss={:.5f}'.format(epoch, i, optimizer.state_dict()['param_groups'][0]['lr'], loss.item()))
        scheduler.step()

        save_path = os.path.join(config.data_path, 'ws/models/model_step{}_{:0>2}.pt'.format(config.active_step, epoch))
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
                print('epoch={:0>3}, batch={:0>4}, lr={:.5f}, loss={:.5f}'.format(epoch, i, optimizer.state_dict()['param_groups'][0]['lr'], loss.item()))
        scheduler.step()

        save_path = os.path.join(config.data_path, 'ws/models/model_step{}_{:0>2}.pt'.format(config.active_step, epoch))
        torch.save(model.state_dict(),  save_path)

    print('training finished')

def camera_demo():
    W = np.load(os.path.join(config.data_path, 'ws/models/W1.npy'))
    landmark_pca_feat = np.load(os.path.join(config.data_path, 'ws/models/landmark_pca_feat.npy'))

    face_detector = get_face_detector()

    pmodel, device = get_pmodel(os.path.join(config.data_path, 'ws/models/model_step1_final.pt'))
    pmodel.eval()
    rmodel, device = get_rmodel(os.path.join(config.data_path, 'ws/models/model_step2_final.pt'), is_train=False)
    rmodel.eval()

    landmark = None
    # camera = cv2.VideoCapture(0)
    camera = cv2.VideoCapture('facetest.mov')
    for _ in range(30):
        ret, image = camera.read()
    image = cv2.transpose(image, (1, 0, 2))
    h, w, _ = image.shape

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    outvideo = cv2.VideoWriter('output.avi', fourcc, 30.0, (w, h))

    while True:
        ret, image = camera.read()
        # ret, image = camera.read()
        image = cv2.transpose(image, (1, 0, 2))
        if ret == 0:
            break

        if landmark is None:
            predout = face_detector.detect_face(image)
            if predout is not None:
                # boxes   = predout[0]
                # det_num = boxes.shape[0]
                points = predout[1]
                p = points[0]
                landmark_5p = [p[0], p[5], p[1], p[6], p[2], p[7], p[3], p[8], p[4], p[9]]
                landmark_5p = np.array(landmark_5p)
                landmark = landmark_5p.dot(W)

        if landmark is not None:
            start_t = time.time()
            if True:
                landmark = reconstruct_landmark_from_pca(landmark, landmark_pca_feat, 1)
                feats = crop_feature(image, landmark, config.step1_scale, config.step1_resize, config.step1_idx, config.step1_crop_size)
                feats = np.concatenate(feats, axis=2)
                feats = feats.transpose(2, 0, 1)[np.newaxis, ...]
                with torch.no_grad():
                    pred = pmodel(torch.from_numpy(feats).to(device, dtype=torch.float))
                    pred = pred.numpy()[0]
                landmarky = landmark[1::2]
                miny = np.min(landmarky)
                maxy = np.max(landmarky)
                landmark = landmark + pred * (maxy - miny) / 100.
            else:
                landmark = reconstruct_landmark_from_pca(landmark, landmark_pca_feat, 12)

            feats = crop_feature(image, landmark, config.step2_scale, config.step2_resize, config.step2_idx, config.step2_crop_size)
            feats = np.concatenate(feats, axis=2)
            feats = feats.transpose(2, 0, 1)[np.newaxis, ...]

            with torch.no_grad():
                pred, isface = rmodel(torch.from_numpy(feats).to(device, dtype=torch.float))
                pred = pred.numpy()[0]
                isface = isface.numpy()[0]
            conf = isface[1]
            print('{:.3f}  ==>  conf:{:.5f}'.format(time.time() - start_t, conf))

            if conf > 0.8:
                landmarky = landmark[1::2]
                miny = np.min(landmarky)
                maxy = np.max(landmarky)
                landmark = landmark + pred * (maxy - miny) / 100.
                for i in range(config.landmark_num):
                    cv2.circle(image, (int(landmark[2*i]), int(landmark[2*i + 1])), 5, (255,0,0), -1)
            else:
                landmark = None

        outvideo.write(image)
        # cv2.imshow('image', image)
        # if cv2.waitKey(10) & 0xFF == ord('q'):
        #     break

    outvideo.release()
    print('done')


if __name__ == '__main__':
    # test_model()

    # test_rmodel()

    # train_pmodel()
    
    train_rmodel()

    # camera_demo()
