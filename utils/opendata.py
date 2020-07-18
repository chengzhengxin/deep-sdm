import os
import random
import cv2
import math
import pickle
import numpy as np
from sklearn.decomposition import PCA
import util


def convert_data_to_pkl():
    annof = open(os.path.join(config.data_path, 'opendata/anno.txt'), 'r', encoding='utf-8')
    annos = [l.strip() for l in annof.readlines()]
    annof.close()

    image_annos = []
    for anno in annos:
        uints = anno.split(' ')
        ipath = os.path.join('opendata', uints[0])
        landmark = [float(uints[i + 5]) for i in range(2 * config.landmark_num)]
        landmark = np.array(landmark)
        landmark = landmark[config.reset_idx]
        image_annos.append((ipath, landmark))

        # img = cv2.imread(os.path.join(config.data_path, ipath))
        # for j in range(config.landmark_num):
        #     cv2.putText(img, str(j), (int(landmark[2*j]), int(landmark[2*j + 1])), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        #     # cv2.circle(img, (int(landmark[2*j]), int(landmark[2*j + 1])), 3, (0,0,255), -1)
        # cv2.imwrite('opendata.jpg', img)
        # cv2.imshow('img', img)
        # if cv2.waitKey(0) & 0xFF == ord('q'):
        #     break
    
    random.shuffle(image_annos)
    print(len(image_annos))
    save_path = os.path.join(config.data_path, 'ws/data_origin.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(image_annos, f)
    exit()

def mirror_image_annos():
    save_path = os.path.join(config.data_path, 'ws/data_origin.pkl')
    with open(save_path, 'rb') as f:
        image_annos = pickle.load(f)
    m_image_annos = []
    for i_a in image_annos:
        fpath = i_a[0]
        landmarks = i_a[1]
        image = cv2.imread(os.path.join(config.data_path, fpath))
        width = image.shape[1]
        m_image_annos.append((fpath, landmarks.copy(), False))
        # mirror
        landmarks[0::2] = width - 1 - landmarks[0::2]
        landmarkx = landmarks[0::2][config.mirror_idx]
        landmarky = landmarks[1::2][config.mirror_idx]
        landmarks[0::2] = landmarkx
        landmarks[1::2] = landmarky
        m_image_annos.append((fpath, landmarks.copy(), True))
    
    save_path = os.path.join(config.data_path, 'ws/data_mirror.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(m_image_annos, f)
    exit()

def visi_image_annos():
    all_samples, all_landmarks, all_mirrors = util.load_all_datas()

    sample_num = all_landmarks.shape[0]
    for i in range(1000, sample_num):
        landmark = all_landmarks[i]
        img = cv2.imread(os.path.join(config.data_path, all_samples[i]))
        if all_mirrors[i]:
            img = cv2.flip(img, 1)
        for j in range(config.landmark_num):
            # cv2.putText(img, str(j), (int(landmark[2*j]), int(landmark[2*j + 1])), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,255), 1)
            cv2.circle(img, (int(landmark[2*j]), int(landmark[2*j + 1])), 3, (0,0,255), -1)
        cv2.imshow('img', img)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
    exit()

if __name__ == '__main__':
    '''
    按顺序来，你懂的
    '''
    print('start')

    # step 1
    # convert_data_to_pkl()

    # step 2
    # mirror_image_annos()

    # for check & debug
    # visi_image_annos()

    all_samples, all_landmarks, all_mirrors = util.load_all_datas()

    # for check & debug
    # util.visi_all_landmark(all_landmarks)
    util.visi_roi_size(all_landmarks)

    # step 3
    # util.generate_W1(all_landmarks)

    # step 4
    # util.landmark_pca(all_landmarks)

    # step 5
    # util.generate_proposal(all_samples, all_landmarks, all_mirrors)

    # step 6
    # util.generate_train_data_with_crop()

    # step 7
    # run train.py

    # step 8
    # util.generate_proposal_step2_from_step1()

    # step 9: change config file active_step to 2
    # util.generate_proposal(all_samples, all_landmarks, all_mirrors)

    # step 10
    # util.generate_train_data_with_crop()

    # step 11
    # run train.py

