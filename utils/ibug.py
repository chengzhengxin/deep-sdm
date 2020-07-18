import os
import random
import cv2
import math
import pickle
import numpy as np
from sklearn.decomposition import PCA
import util
import config_ibug as config


def convert_ibug_to_pkl():
    def scan_images(data_path=''):
        image_annos = []
        dpath = os.path.join(config.data_path, data_path)
        flist = os.listdir(dpath)
        for f in flist:
            fp1 = os.path.join(dpath, f)
            fp2 = os.path.join(data_path, f)
            if os.path.isdir(fp1):
                l = scan_images(fp2)
                image_annos.extend(l)
            else:
                if f.endswith('.jpg') or f.endswith('.png'):
                    with open(os.path.join(dpath, f[:-3] + 'pts'), 'r') as f:
                        annos = f.readlines()
                    annos = annos[3:-1]
                    pts = []
                    for i in range(len(annos)):
                        pt = annos[i].strip().split(' ')
                        pts.extend([float(pt[0]), float(pt[1])])
                    pts = np.array(pts)
                    image_annos.append((fp2, pts))
        return image_annos
    
    image_annos = scan_images()
    random.shuffle(image_annos)
    print(len(image_annos))
    save_path = os.path.join(config.data_path, 'ws/ibug_data.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(image_annos, f)
    exit()

def mirror_image_annos():
    save_path = os.path.join(config.data_path, 'ws/ibug_data.pkl')
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
    
    save_path = os.path.join(config.data_path, 'ws/ibug_data_mirror.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(m_image_annos, f)
    exit()

def visi_image_annos():
    all_samples, all_landmarks, all_mirrors = util.load_all_ibug_datas()

    sample_num = all_landmarks.shape[0]
    for i in range(sample_num):
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
    # convert_ibug_to_pkl()

    # step 2
    # mirror_image_annos()

    # for check & debug
    # visi_image_annos()

    all_samples, all_landmarks, all_mirrors = util.load_all_ibug_datas()

    # for check & debug
    # util.visi_all_landmark(all_landmarks)
    # util.visi_roi_size(all_landmarks)

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
    util.generate_train_data_with_crop()

    # step 11
    # run train.py

