# -*-utf-8-*-
import numpy as np
import operator
from math import pow,sqrt

class KMeans(object):
	'''Kmeans Cluster Algorithm'''
	
	def __init__(self,K,max_iterations=1000,rate=0.1):
		'''
		params:
			K: cluster center numbers
			max_iterations: max times for stop iterations
			rate: threshold for centers adjustment 
		'''
		self.K = K
		self.max_iterations = max_iterations
		self.rate = rate
	def disEclud(vectA,vectB):
		'''
		calculate distances betwwen vectA and vectB
		'''
		return np.sqrt(np.sum(np.power(vectA-vectB,2)))

	def randCent(dataSet,k):
		n = dataSet.shape[1]

	def fit(self,dataSet,disMeas=disEclud,createCent=randCent):
		'''
		
		'''
		m = dataSet.shape[0]
		if self.K < m:
			raise "k can't be less of the number of sample"
		clusterAssment = np.zeros((m,2))
		centers = init_cluster_center(X,self.K)
		clusterChanged = True
		while clusterChanged and i < max_iterations:
			clusterChanged = False
			for i in range(m):
				minDis = np.inf, minIndex = -1
				for j in range(k):
					disIJ = disMeas(dataSet[i,:],centers[j,:])
					if disIJ < minDis and j > minIndex:
						minDis = disIJ
						minIndex = j
				if clusterAssment[i,0] != minIndex:
					clusterChanged = True
				clusterAssment[i,:] = minIndex,minDis
			for cent in range(k):
				ptsIncluster = dataSet[np.nonzero(clusterAssment[:,0].A==cent)[0]]
				centers[cent,:] = np.mean(ptsIncluster,axis=0)
			print("----class centers")
			print(centers)
		return centers,clusterAssment

	def predict(X):
		pass
	def init_cluster_center(dataSet,k):
		'''
		function:
			init cluster's center
		params:
			dataSet: train data
			K: cluster numbers
		'''
		n = dataSet.shape[1]
		centers = np.zeros((k,n))
		for i in range(n):
			mindata = np.min(dataSet[:,i])
			data_range = np.max(dataSet[:,i]) - mindata
			centers[:,i] = mindata + data_range * np.random.rand(k,1)
		return centers
			