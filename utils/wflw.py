import os
import random
import cv2
import math
import pickle
import numpy as np
from sklearn.decomposition import PCA
from mtcnn_face import *
import config_wflw as config

def load_origin_data(filename):
    anno_file = open(filename, 'r')
    anno_lines = anno_file.readlines()
    anno_file.close()
    random.shuffle(anno_lines)

    all_landmarks = []
    all_boxes = []
    all_samples = []
    all_info = []
    for anno in anno_lines:
        uints = anno.strip().split(' ')
        k = 2 * config.landmark_num
        landmark = [float(uints[i]) for i in range(k)]
        temp = [int(uints[k+i]) for i in range(10)]
        box  = temp[0:4]
        info = temp[4:]
        all_landmarks.append(landmark)
        all_boxes.append(box)
        all_samples.append(uints[-1])
        all_info.append(info)
        # image = cv2.imread(os.path.join(config.data_path, 'WFLW_images', uints[-1]))
        # cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0,0,255), 2)
        # for i in range(config.landmark_num):
        #     cv2.circle(image, (int(landmark[2*i]), int(landmark[2*i + 1])), 2, (0,0,255), -1)
        # cv2.imshow('main', image)
        # cv2.waitKey(0)
    all_landmarks = np.array(all_landmarks)
    all_boxes = np.array(all_boxes)
    all_info = np.array(all_info)
    return all_landmarks, all_boxes, all_info, all_samples

def convert_origin_to_pkl():
    filename = config.data_path + 'WFLW_annotations/list_98pt_rect_attr_train_test/list_98pt_rect_attr_train.txt'
    all_landmarks, all_boxes, all_info, all_samples = load_origin_data(filename)
    with open('train_data.pkl', 'wb') as f:
        pickle.dump((all_landmarks, all_boxes, all_info, all_samples), f)
    
    filename  = config.data_path + 'WFLW_annotations/list_98pt_rect_attr_train_test/list_98pt_rect_attr_test.txt'
    all_landmarks, all_boxes, all_info, all_samples = load_origin_data(filename)
    with open('test_data.pkl', 'wb') as f:
        pickle.dump((all_landmarks, all_boxes, all_info, all_samples), f)
    exit()

def load_pkl_data(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    if len(data) == 4:
        return data[0], data[1], data[2], data[3]
    elif len(data) == 5:
        return data[0], data[1], data[2], data[3], data[4]
    else:
        return None

def generate_random_proposal(all_landmarks, all_boxes, landmark_pca_feat, step=1):
    p = config.step1_proposal if step == 1 else config.step2_proposal
    m = config.step1_m if step == 1 else config.step2_m
    t = 2 if step == 1 else 2

    all_proposal = []
    sample_num = all_landmarks.shape[0]
    for i in range(sample_num):
        box = all_boxes[i, :]
        landmark = all_landmarks[i, :].copy()
        for j in p:
            mlandmark = reconstruct_landmark_from_pca(landmark, landmark_pca_feat, j)
            all_proposal.append((i, mlandmark))

        for _ in range(t):
            landmark = all_landmarks[i, :].copy()
            M = get_random_m(box, m[0], m[1], m[2])
            apply_similar_transform(landmark, M)
            for j in p:
                mlandmark = reconstruct_landmark_from_pca(landmark, landmark_pca_feat, j)
                all_proposal.append((i, mlandmark))
                # debug
                # visi_landmark(all_landmarks[i, :], landmark2=mlandmark)

    return all_proposal

def generate_proposal(landmarks_data, landmark_pca_feat, step=1, landmarks_5p_data=None, W=None, step1_pred_landmarks_data=None):
    all_landmarks, all_boxes, all_info, all_samples = load_pkl_data(landmarks_data)
    all_proposal = generate_random_proposal(all_landmarks, all_boxes, landmark_pca_feat, step)
    if step == 1:
        all_landmarks_5p = np.load(landmarks_5p_data)
        all_landmarks_5p = all_landmarks_5p.dot(W)
        print(all_landmarks_5p.shape)
        sample_num = all_landmarks_5p.shape[0]
        for i in range(sample_num):
            all_proposal.append((i, all_landmarks_5p[i]))
    elif step == 2:
        pass
    else:
        print('ERROR')

    print('proposal num: {}'.format(len(all_proposal)))
    save_path = os.path.join(config.data_path, 'train_data_with_proposal_step{}.pkl'.format(step))
    with open(save_path, 'wb') as f:
        pickle.dump((all_landmarks, all_boxes, all_info, all_samples, all_proposal), f)
    exit()


def box_iou(box1, box2):
    """
    :param box1: = [xmin1, ymin1, xmax1, ymax1]
    :param box2: = [xmin2, ymin2, xmax2, ymax2]
    """
    xmin1, ymin1, xmax1, ymax1 = box1
    xmin2, ymin2, xmax2, ymax2 = box2
    s1 = (xmax1 - xmin1) * (ymax1 - ymin1)  # C的面积
    s2 = (xmax2 - xmin2) * (ymax2 - ymin2)  # G的面积
 
    xmin = max(xmin1, xmin2)
    ymin = max(ymin1, ymin2)
    xmax = min(xmax1, xmax2)
    ymax = min(ymax1, ymax2)
 
    w = max(0, xmax - xmin)
    h = max(0, ymax - ymin)
    area = w * h
    iou = area / (s1 + s2 - area)
    return iou

def generate_all_landmarks_5p(all_landmarks, all_boxes, all_samples):
    all_landmarks_5p = []
    sample_num = all_landmarks.shape[0]
    for i in range(sample_num):
        box1 = all_boxes[i, :]
        landmark_5p = all_landmarks[i, config.idx_5p]
        image = cv2.imread(os.path.join(config.data_path, 'WFLW_images', all_samples[i]))

        predout = det_face(image)
        if predout is not None and len(predout) > 0:
            boxes  = predout[0]
            det_num = boxes.shape[0]
            max_iou = 0.
            max_iou_idx = 0
            for j in range(det_num):
                box2 = boxes[j][:-1]
                iou = box_iou(box1, box2)
                if iou > max_iou:
                    max_iou = iou
                    max_iou_idx = j
            if max_iou > 0.5:
                points = predout[1]
                p = points[max_iou_idx]
                landmark_5p = [p[0], p[5], p[1], p[6], p[2], p[7], p[3], p[8], p[4], p[9]]
                landmark_5p = np.array(landmark_5p)
        all_landmarks_5p.append(landmark_5p)
    all_landmarks_5p = np.array(all_landmarks_5p)
    print(all_landmarks_5p.shape)
    np.save('landmarks_5p.npy', all_landmarks_5p)
    exit()

def crop_image(image, box):
    minx, maxx, miny, maxy = box
    h, w, c = image.shape
    if minx >= 0 and miny >=0 and maxx < w and maxy < h:
        roi_image = image[miny:maxy, minx:maxx, :].copy()
    else:
        pad_size = 200
        pad_size = max(pad_size, abs(minx))
        pad_size = max(pad_size, abs(miny))
        pad_size = max(pad_size, abs(maxx - w))
        pad_size = max(pad_size, abs(maxy - h))
        image = cv2.copyMakeBorder(image, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_CONSTANT, value = (0,0,0))
        minx += pad_size
        maxx += pad_size
        miny += pad_size
        maxy += pad_size
        roi_image = image[miny:maxy, minx:maxx, :].copy()
    return roi_image

def generate_train_data_with_crop(data_with_proposal, step=1):
    scale = config.step1_scale if step == 1 else config.step2_scale
    resize = config.step1_resize if step == 1 else config.step2_resize
    crop_size = config.step1_crop_size if step == 1 else config.step2_crop_size
    sample_idx = config.step1_idx if step == 1 else config.step2_idx

    all_landmarks, all_boxes, all_info, all_samples, all_proposal = load_pkl_data(data_with_proposal)
    data_with_crop = []
    for i in range(len(all_proposal)):
        proposal = all_proposal[i]
        idx = proposal[0]
        box = all_boxes[idx, :]
        target_landmark = all_landmarks[idx, :]
        proposal_landmark = proposal[1]
        image = cv2.imread(os.path.join(config.data_path, 'WFLW_images', all_samples[idx]))

        landmarkx = proposal_landmark[0::2]
        landmarky = proposal_landmark[1::2]
        minx = np.min(landmarkx)
        maxx = np.max(landmarkx)
        miny = np.min(landmarky)
        maxy = np.max(landmarky)
        learn_target = (target_landmark - proposal_landmark) * 100.0 / (maxy - miny)

        ctx = (minx + maxx) / 2
        cty = (miny + maxy) / 2
        csize = int(round((maxy - miny) * scale))
        minx = int(round(ctx - csize / 2))
        miny = int(round(cty - csize / 2))
        roi_image = crop_image(image, (minx, minx + csize, miny, miny + csize))
        roi_image = cv2.resize(roi_image, (resize, resize))

        landmarkx = (landmarkx - minx) * resize / csize
        landmarky = (landmarky - miny) * resize / csize
        feats = []
        for j in sample_idx:
            ctx = landmarkx[j]
            cty = landmarky[j]
            minx = int(round(ctx - crop_size / 2))
            miny = int(round(cty - crop_size / 2))
            feat = crop_image(roi_image, (minx, minx + crop_size, miny, miny + crop_size))
            feats.append(feat)
        feats = np.concatenate(feats, axis=2)
        data_with_crop.append((i, feats, proposal_landmark, target_landmark, learn_target, 1))

        # for i in range(config.landmark_num):
        #     cv2.circle(image, (int(proposal_landmark[2*i]), int(proposal_landmark[2*i + 1])), 2, (0,0,255), -1)
        # feats = np.concatenate(feats, axis=1)
        # cv2.imshow('feats', feats)
        # cv2.imshow('image', image)
        # cv2.imshow('roi_image', roi_image)
        # cv2.waitKey(0)

    if step == 2:
        pass
    print('train_sample_num = {}'.format(len(data_with_crop)))
    sava_path = os.path.join(config.data_path, 'data_with_crop_step{}.pkl'.format(step))
    with open(sava_path, 'wb') as f:
        pickle.dump(data_with_crop, f)
    exit()

if __name__ == '__main__':
    print('runing')
    # convert_origin_to_pkl()

    W = np.load('models/W1.npy')
    landmark_pca_feat = np.load('models/landmark_pca_feat.npy')

    # all_landmarks, all_boxes, all_info, all_samples = load_pkl_data('train_data.pkl')
    # all_landmarks, all_boxes, all_info, all_samples = load_pkl_data('test_data.pkl')

    # debug
    # visi_all_landmark(all_landmarks)
    # visi_roi_size(all_landmarks)


    # step 1
    # generate_W1(all_landmarks, all_boxes)

    # step 2
    # landmark_pca(all_landmarks, all_boxes)

    # step 3
    # generate_all_landmarks_5p(all_landmarks, all_boxes, all_samples)

    # step 4
    # generate_proposal('train_data.pkl', landmark_pca_feat, step=1, landmarks_5p_data='models/train_landmarks_5p.npy', W=W)

    # step 5
    generate_train_data_with_crop(config.data_path + 'train_data_with_proposal_step1.pkl', step=1)

    # all_landmarks, all_boxes, all_info, all_samples, all_proposal = load_pkl_data('train_data_with_proposal_step1.pkl')

    # for p in all_proposal:
    #     i = p[0]
    #     box = all_boxes[i, :]
    #     landmark1 = all_landmarks[i, :]
    #     landmark2 = p[1]
    #     image = cv2.imread(os.path.join(config.data_path, 'WFLW_images', all_samples[i]))
    #     for i in range(config.landmark_num):
    #         cv2.circle(image, (int(landmark1[2*i]), int(landmark1[2*i + 1])), 2, (0,0,255), -1)
    #         cv2.circle(image, (int(landmark2[2*i]), int(landmark2[2*i + 1])), 2, (255,0,0), -1)
    #     cv2.imshow('main', image)
    #     cv2.waitKey(0)

