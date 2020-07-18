import os
import random
import cv2
import math
import pickle
import h5py
import numpy as np
from sklearn.decomposition import PCA
from mtcnn_face import *
# import config.config_wflw as config
# import config.config_ibug as config
import config.config_opendata as config
from model import *


def load_all_datas():
    save_path = os.path.join(config.data_path, 'ws/data_mirror.pkl')
    with open(save_path, 'rb') as f:
        image_annos = pickle.load(f)
    sample_num = len(image_annos)
    all_samples   = [image_annos[i][0] for i in range(sample_num)]
    all_landmarks = [image_annos[i][1] for i in range(sample_num)]
    all_mirrors   = [image_annos[i][2] for i in range(sample_num)]
    all_landmarks = np.array(all_landmarks)
    return all_samples, all_landmarks, all_mirrors

def landmark_to_box(landmark):
    landmarkx = landmark[0::2]
    landmarky = landmark[1::2]
    minx = np.min(landmarkx)
    maxx = np.max(landmarkx)
    miny = np.min(landmarky)
    maxy = np.max(landmarky)
    return (minx, maxx, miny, maxy)

# 获取随机相似变换矩阵
# 随机缩放s、旋转a、偏移t
# 用于数据增强，训练数据的随机扰动
def get_random_m(landmark, s=0.3, a=30, t=0.3):
    minx, maxx, miny, maxy = landmark_to_box(landmark)

    ofx = (maxy - miny) * random.uniform(-t, t)
    ofy = (maxy - miny) * random.uniform(-t, t)
    ctx = (maxx + minx) / 2 + ofx
    cty = (maxy + miny) / 2 + ofy
    scale = 1.0 + random.uniform(-s, s)
    angle = random.uniform(-a, a) * math.pi / 180.0

    alpha = scale * math.cos(angle)
    beta  = scale * math.sin(angle)
    M = [ [alpha, beta, (1.0-alpha)*ctx - beta*cty], [-beta, alpha, beta*ctx + (1.0-alpha)*cty] ]
    M = np.array(M)

    return M

# 应用相似变换
def apply_similar_transform(landmark, M):
    landmarkx = landmark[0::2]
    landmarky = landmark[1::2]
    landmarkm = np.array([landmarkx, landmarky, np.ones_like(landmarkx)])
    landmarkm = landmarkm.T.dot(M.T).T
    landmark[0::2] = landmarkm[0]
    landmark[1::2] = landmarkm[1]

# 线性拟合一下子
def linear_fit(A, C):
    # A * W = C
    # W = (A.T * A)^(-1) * A.T * C
    B = A.T.dot(C)
    D = np.linalg.inv(A.T.dot(A))
    W = D.dot(B)
    return W

def ibug_pick_all_landmarks_5p(all_landmarks):
    i5p = config.mtcnn_idx
    all_landmarks_5p = []
    sample_num = all_landmarks.shape[0]
    for i in range(sample_num):
        landmark = all_landmarks[i, :]
        p = [0. for _ in range(10)]
        p[0] = (landmark[2 * i5p[0][0]] + landmark[2 * i5p[0][1]]) / 2
        p[1] = (landmark[2 * i5p[0][0] + 1] + landmark[2 * i5p[0][1] + 1]) / 2
        p[2] = (landmark[2 * i5p[1][0]] + landmark[2 * i5p[1][1]]) / 2
        p[3] = (landmark[2 * i5p[1][0] + 1] + landmark[2 * i5p[1][1] + 1]) / 2
        p[4] = landmark[2 * i5p[2]]
        p[5] = landmark[2 * i5p[2] + 1]
        p[6] = landmark[2 * i5p[3]]
        p[7] = landmark[2 * i5p[3] + 1]
        p[8] = landmark[2 * i5p[4]]
        p[9] = landmark[2 * i5p[4] + 1]
        all_landmarks_5p.append(np.array(p))
    all_landmarks_5p = np.array(all_landmarks_5p)
    return all_landmarks_5p

def opendata_pick_all_landmarks_5p(all_landmarks):
    i5p = config.mtcnn_idx
    all_landmarks_5p = []
    sample_num = all_landmarks.shape[0]
    for i in range(sample_num):
        landmark = all_landmarks[i, :]
        p = [0. for _ in range(10)]
        p[0] = landmark[2 * i5p[0]]
        p[1] = landmark[2 * i5p[0] + 1]
        p[2] = landmark[2 * i5p[1]]
        p[3] = landmark[2 * i5p[1] + 1]
        p[4] = landmark[2 * i5p[2]]
        p[5] = landmark[2 * i5p[2] + 1]
        p[6] = landmark[2 * i5p[3]]
        p[7] = landmark[2 * i5p[3] + 1]
        p[8] = landmark[2 * i5p[4]]
        p[9] = landmark[2 * i5p[4] + 1]
        all_landmarks_5p.append(np.array(p))
    all_landmarks_5p = np.array(all_landmarks_5p)
    return all_landmarks_5p

def generate_W1(all_landmarks):

    all_landmarks_ = []
    sample_num = all_landmarks.shape[0]
    for i in range(sample_num):
        landmark = all_landmarks[i, :].copy()
        M = get_random_m(landmark, 0.3, 30, 0.15)
        apply_similar_transform(landmark, M)
        all_landmarks_.append(landmark)
    all_landmarks_ = np.array(all_landmarks_)
    all_landmarks = np.vstack((all_landmarks, all_landmarks_))

    all_landmarks_5p = opendata_pick_all_landmarks_5p(all_landmarks)
    # all_landmarks_5p = all_landmarks[:, config.idx_5p]
    W = linear_fit(all_landmarks_5p, all_landmarks)
    print(W.shape)
    np.save(os.path.join(config.data_path, 'ws/models/W1.npy'), W)
    exit()

def landmark_pca(all_landmarks):

    m = config.step1_m
    all_landmarks_ = []
    sample_num = all_landmarks.shape[0]
    for i in range(sample_num):
        landmark = all_landmarks[i, :].copy()
        M = get_random_m(landmark, m[0], m[1], m[2])
        apply_similar_transform(landmark, M)
        all_landmarks_.append(landmark)
    all_landmarks_ = np.array(all_landmarks_)
    all_landmarks = np.vstack((all_landmarks, all_landmarks_))
    
    # norm landmark
    sample_num = all_landmarks.shape[0]
    for i in range(sample_num):
        landmark = all_landmarks[i, :]
        minx, maxx, miny, maxy = landmark_to_box(landmark)
        landmark[0::2] = (landmark[0::2] - minx) / (maxy - miny)
        landmark[1::2] = (landmark[1::2] - miny) / (maxy - miny)
        all_landmarks[i, :] = landmark
    # PCA
    mPCA = PCA(n_components=config.n_components)
    newdata = mPCA.fit_transform(all_landmarks)
    landmark_mean = mPCA.mean_.reshape((1, mPCA.mean_.shape[0]))

    print(mPCA.explained_variance_ratio_)
    print(landmark_mean.shape)
    print(mPCA.components_.shape)

    landmark_pca_feat = np.vstack((landmark_mean, mPCA.components_))
    print(landmark_pca_feat.shape)
    np.save(os.path.join(config.data_path, 'ws/models/landmark_pca_feat.npy'), landmark_pca_feat)
    exit()


def reconstruct_landmark_from_pca(landmark, landmark_pca_feat, n_components=config.n_components):
    minx, maxx, miny, maxy = landmark_to_box(landmark)
    mlandmark = landmark_pca_feat[0].copy()
    if n_components > 0:
        mlandmark[0::2] = (landmark[0::2] - minx) / (maxy - miny)
        mlandmark[1::2] = (landmark[1::2] - miny) / (maxy - miny)
        mlandmark = mlandmark - landmark_pca_feat[0]
        lamda = mlandmark.dot(landmark_pca_feat[1:1+n_components].T)
        mlandmark = landmark_pca_feat[1:1+n_components].T * lamda
        mlandmark = landmark_pca_feat[0] +  mlandmark.T.sum(axis=0)

    mlandmark[0::2] = mlandmark[0::2] * (maxy - miny) + minx
    mlandmark[1::2] = mlandmark[1::2] * (maxy - miny) + miny
    return mlandmark

# for debug
def visi_landmark(landmark, landmark2=None):
    minx, maxx, miny, maxy = landmark_to_box(landmark)
    landmarkx = (landmark[0::2] - minx) / (maxy - miny) * 400 + 50
    landmarky = (landmark[1::2] - miny) / (maxy - miny) * 400 + 50
    image = np.zeros((500, 500, 3), dtype=np.uint8)
    for i in range(config.landmark_num):
        cv2.circle(image, (int(landmarkx[i]), int(landmarky[i])), 4, (0,0,255), -1)
    if landmark2 is not None:
        landmarkx = (landmark2[0::2] - minx) / (maxy - miny) * 400 + 50
        landmarky = (landmark2[1::2] - miny) / (maxy - miny) * 400 + 50
        for i in range(config.landmark_num):
            cv2.circle(image, (int(landmarkx[i]), int(landmarky[i])), 4, (255,0,0), -1)
    cv2.imshow('main', image)
    cv2.waitKey(0)

# for debug
def visi_all_landmark(all_landmarks):
    landmark_pca_feat = np.load(os.path.join(config.data_path, 'ws/models/landmark_pca_feat.npy'))
    proposal = config.step1_proposal if config.active_step == 1 else config.step2_proposal

    sample_num = all_landmarks.shape[0]
    for i in range(sample_num):
        landmark = all_landmarks[i, :]
        for j in proposal:
            mlandmark = reconstruct_landmark_from_pca(landmark, landmark_pca_feat, j)
            visi_landmark(landmark, landmark2=mlandmark)
    exit()

# for debug
def visi_roi_size(all_landmarks):
    sample_num = all_landmarks.shape[0]
    for i in range(sample_num):
        landmark = all_landmarks[i, :]
        minx, maxx, miny, maxy = landmark_to_box(landmark)
        landmarkx = (landmark[0::2] - minx) / (maxy - miny) * 400 + 50
        landmarky = (landmark[1::2] - miny) / (maxy - miny) * 400 + 50
        image = np.zeros((500, 500, 3), dtype=np.uint8)
        for i in range(config.landmark_num):
            cv2.circle(image, (int(landmarkx[i]), int(landmarky[i])), 4, (0,0,255), -1)
        if config.active_step == 1:
            size = config.step1_crop_size * 400 * config.step1_scale / config.step1_resize / 2
        elif config.active_step == 2:
            size = config.step2_crop_size * 400 * config.step2_scale / config.step2_resize / 2
        else:
            pass
        t_idx = config.step1_idx if config.active_step == 1 else config.step2_idx
        for i in t_idx:
            cv2.rectangle(image, (int(landmarkx[i]-size), int(landmarky[i]-size)), \
                                 (int(landmarkx[i]+size), int(landmarky[i]+size)), (255,0,0), 2)
        cv2.imshow('main', image)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
    exit()

def generate_random_proposal(all_landmarks, landmark_pca_feat):
    p = config.step1_proposal if config.active_step == 1 else config.step2_proposal
    m = config.step1_m if config.active_step == 1 else config.step2_m
    t = config.step1_t if config.active_step == 1 else config.step2_t

    all_proposal = []
    sample_num = all_landmarks.shape[0]
    for i in range(sample_num):
        landmark = all_landmarks[i, :].copy()
        for j in p:
            mlandmark = reconstruct_landmark_from_pca(landmark, landmark_pca_feat, j)
            all_proposal.append((i, mlandmark))

        for _ in range(t):
            landmark = all_landmarks[i, :].copy()
            M = get_random_m(landmark, m[0], m[1], m[2])
            apply_similar_transform(landmark, M)
            for j in p:
                mlandmark = reconstruct_landmark_from_pca(landmark, landmark_pca_feat, j)
                all_proposal.append((i, mlandmark))
                # debug
                # visi_landmark(all_landmarks[i, :], landmark2=mlandmark)
    return all_proposal

def generate_proposal(all_samples, all_landmarks, all_mirrors):
    W = np.load(os.path.join(config.data_path, 'ws/models/W1.npy'))
    landmark_pca_feat = np.load(os.path.join(config.data_path, 'ws/models/landmark_pca_feat.npy'))
    all_proposal = generate_random_proposal(all_landmarks, landmark_pca_feat)

    # if config.active_step == 1:
    #     all_landmarks_5p = None
    #     all_landmarks_5p = np.load(landmarks_5p_data)
    #     all_landmarks_5p = all_landmarks_5p.dot(W)
    #     print(all_landmarks_5p.shape)
    #     sample_num = all_landmarks_5p.shape[0]
    #     for i in range(sample_num):
    #         all_proposal.append((i, all_landmarks_5p[i]))
    # elif config.active_step == 2:
    #     pass
    # else:
    #     print('ERROR')

    print('proposal num: {}'.format(len(all_proposal)))
    save_path = os.path.join(config.data_path, 'ws/train_data_with_proposal_step{}.pkl'.format(config.active_step))
    with open(save_path, 'wb') as f:
        pickle.dump((all_samples, all_landmarks, all_mirrors, all_proposal), f)
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

def crop_feature(image, landmark, scale, resize, sample_idx, crop_size):
    minx, maxx, miny, maxy = landmark_to_box(landmark)
    ctx = (minx + maxx) / 2
    cty = (miny + maxy) / 2
    csize = int(round((maxy - miny) * scale))
    minx = int(round(ctx - csize / 2))
    miny = int(round(cty - csize / 2))
    roi_image = crop_image(image, (minx, minx + csize, miny, miny + csize))
    roi_image = cv2.resize(roi_image, (resize, resize))

    landmarkx = (landmark[0::2] - minx) * resize / csize
    landmarky = (landmark[1::2] - miny) * resize / csize
    feats = []
    for j in sample_idx:
        ctx = landmarkx[j]
        cty = landmarky[j]
        minx = int(round(ctx - crop_size / 2))
        miny = int(round(cty - crop_size / 2))
        feat = crop_image(roi_image, (minx, minx + crop_size, miny, miny + crop_size))
        feats.append(feat)

    return feats

def generate_train_data_with_crop():
    scale      = config.scale
    resize     = config.resize
    crop_size  = config.crop_size
    sample_idx = config.sample_idx
    feature_size  = crop_size * crop_size * 3 * len(sample_idx)
    landmark_size = config.landmark_num * 2 * 3
    metadata_size = 2

    # generate negative sample list
    noface_list = None
    if config.active_step == 2:
        noface_list = os.listdir(config.noface_data_path)
        noface_list = [n for n in noface_list if n.endswith('.jpg')]

    # sava_path = os.path.join(config.data_path, 'ws/train_data_with_crop_step{}.hdf5'.format(config.active_step))
    sava_path = os.path.join(config.data_path, 'ws/train_data_with_crop_stepxxx.hdf5')
    h5f = h5py.File(sava_path, 'w')
    h5_feature  = h5f.create_dataset('feature',  shape=(1000000, feature_size),  dtype=np.uint8,   maxshape=(None, feature_size),  chunks=(1, feature_size))
    h5_landmark = h5f.create_dataset('landmark', shape=(1000000, landmark_size), dtype=np.float32, maxshape=(None, landmark_size), chunks=(1, landmark_size))
    h5_metadata = h5f.create_dataset('metadata', shape=(1000000, metadata_size), dtype=np.int32,   maxshape=(None, metadata_size), chunks=(1, metadata_size))
    sample_len  = 0

    def push_hdf5_data(feature, landmark, metadata):
        nonlocal sample_len
        h5_feature[sample_len, :]  = feature
        h5_landmark[sample_len, :] = landmark
        h5_metadata[sample_len, :] = metadata
        sample_len += 1

    def generate_crop_feature(all_samples, all_landmarks, all_mirrors, all_proposal):
        nonlocal scale, resize, crop_size, sample_idx
        for i, proposal in enumerate(all_proposal):
            if i % 100 == 0:
                print('{}/{}'.format(i, len(all_proposal)))
            idx = proposal[0]
            proposal_landmark = proposal[1]
            target_landmark = all_landmarks[idx, :]
            image = cv2.imread(os.path.join(config.data_path, all_samples[idx]))
            if all_mirrors[idx]:
                image = cv2.flip(image, 1)

            minx, maxx, miny, maxy = landmark_to_box(proposal_landmark)
            learn_target = (target_landmark - proposal_landmark) * 100.0 / (maxy - miny)
            feats = crop_feature(image, proposal_landmark, scale, resize, sample_idx, crop_size)
            feats = np.concatenate(feats, axis=2)
            feats = feats.flatten()

            landmark = np.concatenate((learn_target, proposal_landmark, target_landmark)).astype(np.float32)
            metadata = np.array([idx, 1], dtype=np.int32)
            push_hdf5_data(feats, landmark, metadata)

            if noface_list is not None and i % 10 == 0:
                img_path  = noface_list[i//10 % len(noface_list)]
                roi_image = cv2.imread(os.path.join(config.noface_data_path, img_path))
                if roi_image is None:
                    continue
                roi_image = cv2.resize(roi_image, (resize, resize))

                ctx = (minx + maxx) / 2
                cty = (miny + maxy) / 2
                csize = int(round((maxy - miny) * scale))
                minx = int(round(ctx - csize / 2))
                miny = int(round(cty - csize / 2))
                landmarkx = (proposal_landmark[0::2] - minx) * resize / csize
                landmarky = (proposal_landmark[1::2] - miny) * resize / csize

                feats = []
                for j in sample_idx:
                    ctx = landmarkx[j]
                    cty = landmarky[j]
                    minx = int(round(ctx - crop_size / 2))
                    miny = int(round(cty - crop_size / 2))
                    feat = crop_image(roi_image, (minx, minx + crop_size, miny, miny + crop_size))
                    feats.append(feat)
                feats = np.concatenate(feats, axis=2)
                feats = feats.flatten()

                landmark = np.zeros((landmark_size,), dtype=np.float32)
                metadata = np.array([-1, 0], dtype=np.int32)
                push_hdf5_data(feats, landmark, metadata)

            # for debug
            # for j in range(config.landmark_num):
            #     cv2.circle(image, (int(target_landmark[2*j]), int(target_landmark[2*j + 1])), 4, (255,0,0), -1)
            #     cv2.circle(image, (int(proposal_landmark[2*j]), int(proposal_landmark[2*j + 1])), 2, (0,0,255), -1)
            # mfeats = []
            # for feat in feats:
            #     feat = cv2.copyMakeBorder(feat, 0, 0, 0, 2, cv2.BORDER_CONSTANT, value = (0,0,0))
            #     mfeats.append(feat)
            # mfeats = np.concatenate(mfeats, axis=1)
            # cv2.imshow('mfeats', mfeats)
            # cv2.imshow('image', image)
            # cv2.imshow('roi_image', roi_image)
            # cv2.waitKey(0)
    
    save_path = os.path.join(config.data_path, 'ws/train_data_with_proposal_step{}.pkl'.format(config.active_step))
    with open(save_path, 'rb') as f:
        fdata = pickle.load(f)
    all_samples, all_landmarks, all_mirrors, all_proposal = fdata[0], fdata[1], fdata[2], fdata[3]
    generate_crop_feature(all_samples, all_landmarks, all_mirrors, all_proposal)

    if config.active_step == 2:
        save_path = os.path.join(config.data_path, 'ws/train_data_with_proposal_step2_from_step1.pkl')
        with open(save_path, 'rb') as f:
            fdata = pickle.load(f)
        all_samples, all_landmarks, all_mirrors, all_proposal = fdata[0], fdata[1], fdata[2], fdata[3]
        generate_crop_feature(all_samples, all_landmarks, all_mirrors, all_proposal)

    print('train_sample_num = {}'.format(sample_len))
    h5_feature.resize((sample_len,  feature_size))
    h5_landmark.resize((sample_len, landmark_size))
    h5_metadata.resize((sample_len, metadata_size))
    h5f.close()
    exit()



def generate_proposal_step2_from_step1():
    model, device = get_pmodel(os.path.join(config.data_path, 'ws/models/model_step1_final.pt'))
    model.eval()

    all_proposal = []
    data_path = os.path.join(config.data_path, 'ws/train_data_with_crop_step1.pkl')
    with open(data_path, 'rb') as f:
        origin_data = pickle.load(f)
    # [(idx, feats, proposal_landmark, target_landmark, learn_target, is_face), ...]
    for i, data in enumerate(origin_data):
        if i % 1000 == 0:
            print('runing  ==>  {}'.format(i))
        sample_idx = data[0]
        input_data = data[1]
        proposal_landmark = data[2]

        input_data = input_data.transpose(2, 0, 1)[np.newaxis, ...]
        with torch.no_grad():
            pred = model(torch.from_numpy(input_data).to(device, dtype=torch.float))
        pred = pred.numpy()[0]
        landmarky = proposal_landmark[1::2]
        miny = np.min(landmarky)
        maxy = np.max(landmarky)
        pred_landmark = proposal_landmark + pred * (maxy - miny) / 100.
        all_proposal.append((sample_idx, pred_landmark))

    print('proposal num: {}'.format(len(all_proposal)))
    
    data_path = os.path.join(config.data_path, 'ws/train_data_with_proposal_step1.pkl')
    with open(data_path, 'rb') as f:
        fdata = pickle.load(f)
    all_samples, all_landmarks, all_mirrors, _ = fdata[0], fdata[1], fdata[2], fdata[3]

    save_path = os.path.join(config.data_path, 'ws/train_data_with_proposal_step2_from_step1.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump((all_samples, all_landmarks, all_mirrors, all_proposal), f)
    exit()
