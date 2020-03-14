from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import numpy as np
from tqdm import tqdm
from PIL import Image


ori_folder = 'datas/KCMI-550'
data_folder = 'results/KCMI-550-Ours'
# ori_folder = 'datas/VISION-1500'
# data_folder = 'results/VISION-1500-Ours'
# ori_folder = 'results/KCMI-550-Crop'
# data_folder = 'results/KCMI-550-Crop-Ours'
paths = os.listdir(ori_folder)

paths_filter = []
for p in paths:
	if 'jpg' in p or 'png' in p:
		paths_filter.append(p)
paths = paths_filter
paths.sort()

l1_record = []
for i in tqdm(range(len(paths))):
	im_ori  = np.asarray(Image.open(os.path.join(ori_folder, paths[i]))).astype(np.float32)
	im_pred = np.asarray(Image.open(os.path.join(data_folder, paths[i][:-3]+'png'))).astype(np.float32)
	im_diff = np.abs(im_ori - im_pred)
	l1 = np.mean(im_diff)
	l1_record.append(l1)
print(np.mean(l1_record))
