import os

landmark_num = 68

# PCA n_components
n_components = 12

data_path  = '/Volumes/DATA/data/landmark_data/68p/ibug解压/'
model_path = 'models/'
noface_data_path = '/Volumes/DATA/data/landmark_data/noface'

mirror_idx = [16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 27, 28, 29, 30, 35, 34, 33, 32, 
              31, 45, 44, 43, 42, 47,46, 39, 38, 37, 36, 41, 40, 54, 53, 52, 51, 50, 49, 48, 59, 58, 57, 56, 55, 64, 63, 62, 61, 60, 67, 66, 65]

mtcnn_idx  = [(37, 40), (43, 46), 30, 48, 54]


######################## step1 ########################
step1_idx = [3, 13, 36, 45, 30, 60, 64]     # 7P
step1_m = (0.2, 20, 0.3)
step1_t = 3
step1_proposal = [0, 1]
step1_scale = 1.5
step1_resize = 80
step1_crop_size = 12


######################## step2 ########################
step2_idx = [1, 3, 5, 7, 9, 11, 13, 15, 18, 20, 23, 25, 36, 37, 39, 42, 43, 45, 28, 30, 32, 34, 48, 50, 52, 54, 65, 67]     # 28P
step2_m = (0.1, 15, 0.15)
step2_t = 3
step2_proposal = [8, 12]
step2_scale = 1.5
step2_resize = 120
step2_crop_size = 10


######################## train ########################
active_step = 2
log_interval = 100
no_cuda = True
seed = 1234
batch_size = 64
# optimizer = 'sgd'
optimizer = 'adam'
learning_rate = 0.001
lr_step = 10
lr_gamma = 0.1
momentum = 0.8
epochs = 30
