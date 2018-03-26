# -*- coding:utf-8 -*-

import numpy as np

class CART(object):
	"""docstring for CART"""
	def __init__(self,tolerance=1,minLeafSize=3):
		super(CART,self).__init__()
		self.tolerance =tolerance
		self.minLeafSize = minLeafSize


	def SplitDataSet(self,dataSet,axis,value):
		index1 = np.nonzero(dataSet[:,axis] <= value)[0]
		index2 = np.nonzero(dataSet[:,axis] > value)[0]
		subDataSet1 = dataSet[index1,:]
		subDataSet2 = dataSet[index2,:]
		return subDataSet1,subDataSet2

	def chooseBestSplit(self,dataSet,tolerance,minLeafSize):
		if len(set(dataSet[-1].T.tolist())) == 1:
			return None, np.mean(dataSet[:,-1])
		m,n = dataSet.shape
		currentEntroy = np.var(dataSet[:,-1]) * m
		bestEntroy = np.inf
		bestIndex = -1
		bestValue = -1
		for index in range(n-1):
			for value in set(dataSet[:,index]):
				subDataSet1,subDataSet2 = self.SplitDataSet(dataSet,index,value)
				if (subDataSet1.shape[0] < minLeafSize) or (subDataSet2.shape[0] < minLeafSize):
					continue
				newEntroy = np.var(subDataSet1[:,-1]) *subDataSet1.shape[0] + np.var(subDataSet2[:,-1]) *subDataSet2.shape[0]
				if newEntroy < bestEntroy:
					bestIndex = index
					bestValue = value
					bestEntroy = newEntroy
		if (currentEntroy - bestEntroy) < tolerance:
			return None, np.mean(dataSet[:,-1])
		return bestIndex,bestValue

	def creatTree(self,dataSet,tolerance=1,minLeafSize=3):
		bestIndex,bestValue = self.chooseBestSplit(dataSet,tolerance,minLeafSize)
		if bestIndex == None:
			return bestValue	
		subDataSet1,subDataSet2 = SplitDataSet(dataSet,bestIndex,bestValue)
		retTree = {}
		retTree['feature'] = bestIndex
		retTree['value'] = bestValue
		retTree['left'] = creatTree(subDataSet1,tolerance,minLeafSize)
		retTree['right'] = creatTree(subDataSet1,tolerance,minLeafSize)
		return tree

	def fit(self,dataSet):
		self.tree = self.creatTree(dataSet,self.tolerance,self.minLeafSize)


dataSet=np.array([[1,1,1],[1,2,3],[1,3,4]])
cart = CART()
d1,d2 = cart.SplitDataSet(dataSet,1,2) 
cart.fit(dataSet)
print(cart.tree)