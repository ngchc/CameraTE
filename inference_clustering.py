import os
import torch
import numpy as np
from time import time
from tqdm import tqdm
from scipy import stats
from sklearn import preprocessing
from sklearn.decomposition import PCA

import sys
sys.path.append('libs')
from libKMCUDA import kmeans_cuda


K = 90		# number of clusters
N = 128		# number of features
P = False	# PCA decomposition
L2 = True	# L2 normalization
R = 1		# repeat iter [1,10]

# fetch features
feature_folder = 'results/VISION-1500-Ours-ResNet'
#feature_folder = 'results/VISION-1500-Ours-DenseNet'
path = os.listdir(feature_folder)
path.sort()
n_samples = len(path)

X = []
X_lb = []
for i in tqdm(range(len(path))):
	tmp = np.load(os.path.join(feature_folder, path[i]))
	tmp = tmp[(R-1)*N:R*N,:]	# fetch N features
	if L2 == True:				# l2-normalize
		tmp = preprocessing.normalize(tmp, norm='l2')
	
	tmp_lb = [int(path[i][:path[i].find('_')])] * tmp.shape[0]
	X.append(tmp)
	X_lb.append(tmp_lb)
X = np.concatenate(X, axis=0)
X_lb = np.concatenate(X_lb, axis=0)

# PCA decomposition
if P == True:
	pca = PCA(n_components=256, svd_solver='auto')
	X = pca.fit_transform(X)
print(X.shape)

# apply kmeans
centroids, k_lb = kmeans_cuda(X, K, tolerance=0.01, init='k-means++', metric='L2', verbosity=1, seed=0)
centroids = torch.Tensor(centroids)

# generate mapping from k_lb to lb
mapping = np.zeros(K)
missed_i = []
for i in range(K):
	tmp = X_lb[k_lb==i]
	if tmp.shape[0] == 0:
		mapping[i] = -1
		missed_i.append(i)
	else:
		mapping[i] = stats.mode(tmp)[0][0]

# cluster
acc_count = 0
for i in tqdm(range(len(path))):
	if P == True:
		x = X[i*N:(i+1)*N]
	else:
		x = np.load(os.path.join(feature_folder, path[i]))
		x = x[(R-1)*N:R*N,:]
		if L2 == True:
			x = preprocessing.normalize(x, norm='l2')
	x_lb = int(path[i][:path[i].find('_')])
	
	preds = []
	for j in range(len(x)):
		with torch.no_grad():
			q = torch.Tensor(x[j:j+1])
			dist = torch.norm(q-centroids, p='fro', dim=1, keepdim=False)
			dist[missed_i] = float('inf')
		preds.append(np.array([int(dist.argmin().numpy())]))

	preds = np.concatenate(preds, axis=0)
	pred = stats.mode(preds)[0][0]
	
	if mapping[pred] == x_lb:
		acc_count += 1

print('ACC: %d/%d = %.2f%s' % (acc_count, n_samples, acc_count/n_samples*100, '%'))
