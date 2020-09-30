import numpy as np
import torch
import h5py


path  = '../data/Charades/data_video.h5'

pre_extracted_features_path = '/home/halamri/data/Charades_v1_features_rgb/'
features_type = 'RGB'

max_ques_count = 10

vid_file = h5py.File(path,'r')
vid_features = torch.from_numpy(np.array(vid_file['images_train']))

vid_feat_size = vid_features.size()
# repeat the video features to be provided for every round:

vid = vid_features.view(-1, 1, vid_feat_size[1])
vid = vid.repeat(1, max_ques_count,1)
vid = vid.view(-1, vid_feat_size[1])



print('Done')

