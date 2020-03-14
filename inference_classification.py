from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import cv2
import torch
import numpy as np
from scipy import stats
from PIL import Image
from time import time

import sys
sys.path.append('models')
from model_resnet import Model
#from model_densenet import Model
from torchvision import transforms


# global settings
CROP_SIZE = 224
TTA_COUNT = 4
suffixes = ['JPG', 'JPEG', 'jpeg']

# restore model
model = Model()
weights_path = 'models/model_resnet.pth'
#weights_path = 'models/model_densenet.pth'
model.load(weights_path)
model.eval()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
if torch.cuda.device_count() > 1 and TTA_COUNT > 1 and TTA_COUNT % 2 == 0:
	model = torch.nn.DataParallel(model)

# inference loop
toP = transforms.ToPILImage()
toT = transforms.ToTensor()
rdcp = transforms.RandomCrop(CROP_SIZE)
norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

num_count = 0
acc_count = 0
folder = 'results/KCMI-550-Ours'
for line in open('datas/KCMI-550-Val'):
	num_count += 1
	
	# handle the suffix issue
	file_path = os.path.join(folder, line.split(' ')[0])
	if not os.path.exists(file_path):
		for suffix in suffixes:
			pos = file_path.rfind('.')
			file_path = file_path[:pos]
			file_path += '.' + suffix
			if os.path.exists(file_path):
				break
	name = file_path[file_path.rfind('/')+1:-4]+'png'
	im = np.asarray(Image.open(os.path.join(folder, name)))
	
	# randomly crop patches
	im_batch = []
	for _ in range(TTA_COUNT):
		im_crop = rdcp(toP(im))
		im_crop = toT(im_crop)
		im_crop = norm(im_crop).data.numpy()
		im_batch.append(np.expand_dims(im_crop, 0))
	im_batch = np.concatenate(im_batch, axis=0)
	im = torch.Tensor(im_batch).to(device)
	
	# parse label
	lb = int(line.strip('\n').split(' ')[1])
	
	with torch.no_grad():
		_, logit = model(im)
	logit = logit.cpu().data.numpy()
	pred = logit.argmax(axis=-1)
	pred = stats.mode(pred)[0][0]
	
	if pred == lb:
		acc_count += 1
	print(num_count, name)

print('ACC:%.2f%s (%d/%d)' % (acc_count/550*100, '%', acc_count, 550))
