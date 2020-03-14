from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from scipy import stats
from PIL import Image

import sys
sys.path.append('models')
from model_resnet import Model
#from model_densenet import Model
from torchvision import transforms


# global settings
CROP_SIZE = 224
TTA_COUNT = 128
REPEAT_ITER = 1
#REPEAT_ITER = 10

# restore model
model = Model()
weights_path = 'models/model_resnet.pth'
#weights_path = 'models/model_densenet.pth'
model.load(weights_path)
model.eval()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
if TTA_COUNT > 1 and TTA_COUNT % 2 == 0:
	model = torch.nn.DataParallel(model)

# pre-processing operators
toP = transforms.ToPILImage()
toT = transforms.ToTensor()
rdcp = transforms.RandomCrop(CROP_SIZE)
norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

data_folder = 'results/VISION-1500-Ours'
cache_folder = 'results/VISION-1500-Ours-ResNet'
#cache_folder = 'results/VISION-1500-Ours-DenseNet'
paths = os.listdir(data_folder)
paths.sort()

# inference loop
for i in tqdm(range(len(paths))):
	im = Image.open(os.path.join(data_folder, paths[i]))
	lb = int(paths[i][1:3]) - 1
	
	feats = []
	for _ in range(REPEAT_ITER):
		im_batch = []
		for _ in range(TTA_COUNT):
			im_crop = rdcp(im)
			im_crop = toT(im_crop)
			im_crop = norm(im_crop).data.numpy()
			im_batch.append(np.expand_dims(im_crop, 0))
		
		im_batch = np.concatenate(im_batch, axis=0)
		im_batch = torch.Tensor(im_batch).to(device)
		feat, _ = model(im_batch)
		feat = feat.cpu().data.numpy()
		feats.append(feat)
	
	feats = np.concatenate(feats, axis=0)
	np.save(os.path.join(cache_folder, str(lb) + '_' + paths[i][:paths[i].find('.')] + '.npy'), feats)
