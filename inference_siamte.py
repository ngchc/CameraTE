from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from collections import OrderedDict

import sys
sys.path.append('models')
from model_ddfn import DDFN


folder = 'datas/KCMI-550'
output_folder = 'results/KCMI-550-Ours'
# folder = 'datas/VISION-1500'
# output_folder = 'results/VISION-1500-Ours'
filepaths = os.listdir(folder)
filepaths.sort()

# filter
valid_files = []
for j in range(len(filepaths)):
	n = filepaths[j]
	if n[n.rfind('.'):] == '.jpg':
		valid_files.append(n)
filepaths = valid_files

# restore model
model = DDFN()
checkpoint = torch.load('models/model_ddfn.pth')

new_state_dict = OrderedDict()
state_dict = checkpoint['model_weights']
for k, v in state_dict.items():
	name = k[7:] # remove module.
	new_state_dict[name] = v
model.load_state_dict(new_state_dict)
model = model.cuda()

# initialize output
stride = 224
in_shape = [336, 336]
out_shape = [224, 224]
pad_shape = [(in_shape[0]-out_shape[0])//2, (in_shape[1]-out_shape[1])//2]

# inference loop
with torch.no_grad():
	for i in tqdm(range(0, len(filepaths))):
		im = np.array(Image.open(os.path.join(folder, filepaths[i])))
		h,w,_ = im.shape
		if h < 1000 or w < 1000:
			input = im
			input = input.astype(np.float32) / 255.0
			input = np.transpose(input, [2, 0, 1])
			input = np.expand_dims(input, axis=0)
			input = torch.Tensor(input).cuda()
			
			pred = model(input).data.cpu().numpy()
			pred = np.squeeze(pred)
			pred = np.transpose(pred, [1, 2, 0])
			pred[pred<0] = 0
			pred[pred>1] = 1
			output = (pred * 255.0).astype(np.uint8)
		else:
			im = np.pad(im, ((pad_shape[0],pad_shape[0]), (pad_shape[1],pad_shape[1]), (0,0)), 'constant', constant_values=0)
			output = np.zeros([h, w, 3], dtype=np.uint8)
			for y in list(np.arange(0, im.shape[0] - in_shape[0], stride)) + [im.shape[0] - in_shape[0]]:
				for x in list(np.arange(0, im.shape[1] - in_shape[1], stride)) + [im.shape[1] - in_shape[1]]:
					input = im[y : y + in_shape[0], x : x + in_shape[1], :]
					input = input.astype(np.float32) / 255.0
					input = np.transpose(input, [2, 0, 1])
					input = np.expand_dims(input, axis=0)
					input = torch.Tensor(input).cuda()
					
					pred = model(input).data.cpu()
					pred = torch.nn.functional.pad(pred, (-pad_shape[1], -pad_shape[1], -pad_shape[0], -pad_shape[0])).numpy()
					pred = np.squeeze(pred)
					pred = np.transpose(pred, [1, 2, 0])
					pred[pred<0] = 0
					pred[pred>1] = 1
					pred = (pred * 255.0).astype(np.uint8)
					output[y : y + out_shape[0], x : x + out_shape[1]] = pred
		
		Image.fromarray(output).save(os.path.join(output_folder, filepaths[i][:filepaths[i].find('.')]+'.png'))
