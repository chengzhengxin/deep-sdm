import os

landmark_num = 98

# PCA n_components
n_components = 12

data_path = '/Volumes/DATA/data/landmark_data/98p/WFLW/'

idx_5p = [96, 97, 54, 76, 82]

######################## step1 ########################
# 7P
step1_idx = [7, 25, 54, 96, 97, 76, 82]
step1_m = (0.2, 20, 0.3)
step1_proposal = [0, 1]
step1_scale = 1.5
step1_resize = 80
step1_crop_size = 12


######################## step2 ########################
# 29P
step2_idx = [2, 5, 8, 11, 14, 16, 18, 21, 24, 27, 30, 33, 35, 38, 46, 44, 50, 53, 57, 60, 96, 64, 72, 97, 68, 88, 90, 92, 85]
step2_m = (0.15, 15, 0.15)
step2_proposal = [12]
step2_scale = 1.5
step2_resize = 100
step2_crop_size = 8


train_step = 1
log_interval = 100
no_cuda = True
seed = 1234
batch_size = 64
optimizer = 'sgd'
# optimizer = 'adam'
learning_rate = 0.001
lr_step = 15
lr_gamma = 0.1
momentum = 0.8
epochs = 50
