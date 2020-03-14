from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class MsFeat(nn.Module):
	def __init__(self, in_channels):
		super(MsFeat, self).__init__()
		self.conv1 = nn.Sequential(nn.Conv2d(in_channels, 16, 3, padding=1, dilation=1, bias=True), nn.ReLU(inplace=True))
		init.kaiming_normal_(self.conv1[0].weight, 0, 'fan_in', 'relu'); init.constant_(self.conv1[0].bias, 0.0)
		
		self.conv2 = nn.Sequential(nn.Conv2d(in_channels, 16, 3, padding=2, dilation=2, bias=True), nn.ReLU(inplace=True))
		init.kaiming_normal_(self.conv2[0].weight, 0, 'fan_in', 'relu'); init.constant_(self.conv2[0].bias, 0.0)
		
		self.conv3 = nn.Sequential(nn.Conv2d(16, 16, 3, padding=1, dilation=1, bias=True), nn.ReLU(inplace=True))
		init.kaiming_normal_(self.conv3[0].weight, 0, 'fan_in', 'relu'); init.constant_(self.conv3[0].bias, 0.0)
		
		self.conv4 = nn.Sequential(nn.Conv2d(16, 16, 3, padding=2, dilation=2, bias=True), nn.ReLU(inplace=True))
		init.kaiming_normal_(self.conv4[0].weight, 0, 'fan_in', 'relu'); init.constant_(self.conv4[0].bias, 0.0)
	
	def forward(self, inputs):
		conv1 = self.conv1(inputs)
		conv2 = self.conv2(inputs)
		conv3 = self.conv3(conv2)
		conv4 = self.conv4(conv1)
		return torch.cat((conv1, conv2, conv3, conv4), 1)


class Block(nn.Module):
	def __init__(self, in_channels):
		super(Block, self).__init__()
		self.conv1 = nn.Sequential(nn.Conv2d(in_channels, 64, 1, padding=0, dilation=1, bias=True), nn.ReLU(inplace=True))
		init.kaiming_normal_(self.conv1[0].weight, 0, 'fan_in', 'relu'); init.constant_(self.conv1[0].bias, 0.0)
		
		self.feat1 = nn.Sequential(nn.Conv2d(64, 16, 3, padding=1, dilation=1, bias=True), nn.ReLU(inplace=True))
		init.kaiming_normal_(self.feat1[0].weight, 0, 'fan_in', 'relu'); init.constant_(self.feat1[0].bias, 0.0)
		
		self.feat15 = nn.Sequential(nn.Conv2d(16, 8, 3, padding=2, dilation=2, bias=True), nn.ReLU(inplace=True))
		init.kaiming_normal_(self.feat15[0].weight, 0, 'fan_in', 'relu'); init.constant_(self.feat15[0].bias, 0.0)
		
		self.feat2 = nn.Sequential(nn.Conv2d(64, 16, 3, padding=2, dilation=2, bias=True), nn.ReLU(inplace=True))
		init.kaiming_normal_(self.feat2[0].weight, 0, 'fan_in', 'relu'); init.constant_(self.feat2[0].bias, 0.0)
		
		self.feat25 = nn.Sequential(nn.Conv2d(16, 8, 3, padding=1, dilation=1, bias=True), nn.ReLU(inplace=True))
		init.kaiming_normal_(self.feat25[0].weight, 0, 'fan_in', 'relu'); init.constant_(self.feat25[0].bias, 0.0)
		
		self.feat = nn.Sequential(nn.Conv2d(48, 16, 1, padding=0, dilation=1, bias=True), nn.ReLU(inplace=True))
		init.kaiming_normal_(self.feat[0].weight, 0, 'fan_in', 'relu'); init.constant_(self.feat[0].bias, 0.0)
		
	def forward(self, inputs):
		conv1 = self.conv1(inputs)
		feat1 = self.feat1(conv1)
		feat15 = self.feat15(feat1)
		feat2 = self.feat2(conv1)
		feat25 = self.feat25(feat2)
		feat = self.feat(torch.cat((feat1, feat15, feat2, feat25), 1))
		return torch.cat((inputs, feat), 1)
	

class DDFN(nn.Module):
	def __init__(self, in_channels=3):
		super(DDFN, self).__init__()
		base = 64
		grow = 16
		self.msfeat = MsFeat(in_channels)
		self.dfus_block0 = Block(base)
		self.dfus_block1 = Block(base + grow * 1)
		self.dfus_block2 = Block(base + grow * 2)
		self.dfus_block3 = Block(base + grow * 3)
		self.dfus_block4 = Block(base + grow * 4)
		self.dfus_block5 = Block(base + grow * 5)
		self.dfus_block6 = Block(base + grow * 6)
		self.dfus_block7 = Block(base + grow * 7)
		self.convr = nn.Conv2d(base + grow * 8, in_channels, 1, padding=0, dilation=1, bias=False)
		init.normal_(self.convr.weight, mean=0.0, std=0.001)
	
	def forward(self, inputs):
		msfeat = self.msfeat(inputs)
		b0 = self.dfus_block0(msfeat)
		b1 = self.dfus_block1(b0)
		b2 = self.dfus_block2(b1)
		b3 = self.dfus_block3(b2)
		b4 = self.dfus_block4(b3)
		b5 = self.dfus_block5(b4)
		b6 = self.dfus_block6(b5)
		b7 = self.dfus_block7(b6)
		convr = self.convr(b7)
		return convr


if __name__ == '__main__':
	""" example of weight sharing """
	#self.convs1_siamese = Conv3x3Stack(1, 12, negative_slope)
	#self.convs1_siamese[0].weight = self.convs1[0].weight
	
	import numpy as np
	model = DDFN().cuda()
	x = torch.tensor(np.random.random((1, 3, 1500, 1000)).astype(np.float32)).cuda()
	with torch.no_grad():
		out = model(x)
	
	print(out.shape)
	print('break')
