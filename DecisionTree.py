# -*- coding:utf-8 -*-
import numpy as np
from math import log
import operator

class DecisionTree(object):
	"""docstring for DecisionTree"""
	def __init__(self):
		super(DecisionTree, self).__init__()

	def shannonEnt(self,dataSet):
		'''
		calculate shannon Ent
		'''
		n = len(dataSet)
		labelConunts = {}
		for featVec in dataSet:
			currentLabel = featVec[-1]
			if currentLabel not in labelConunts.keys():
				labelConunts[currentLabel] = 0
			labelConunts[currentLabel] += 1
		Ent = 0.0
		for key in labelConunts:
			prob = labelConunts[key] / n
			Ent -= prob * log(prob,2)
		return Ent

	def splitDataSet(self,dataSet,axis,value):
		retDataset = []
		for featVec in dataSet:
			if featVec[axis] == value:
				reducedFeatVec = featVec[:axis]
				reducedFeatVec.extend(featVec[axis+1:])
				retDataset.append(reducedFeatVec)
		
		return retDataset

	def chooseBestFeatureTosplit(self,dataSet):
		numFeatures = len(dataSet[0]) - 1
		baseEntropy = self.shannonEnt(dataSet)
		bestInfoGain = 0
		bestFeature = -1
		for i in range(numFeatures):
			featList = [sample[i] for sample in dataSet]
			featSet = set(featList)
			newEntropy = 0
			for value in featSet:
				subDataSet = self.splitDataSet(dataSet,i,value)
				prob = len(subDataSet) / len(dataSet)
				newEntropy += prob * self.shannonEnt(subDataSet)
			infoGain = baseEntropy - newEntropy
			if infoGain > bestInfoGain:
				bestInfoGain = infoGain
				bestFeature = i
		return bestFeature

	def majorityCnt(self,classList):
		classCount = {}
		for c in classList:
			if c not in classCount.keys():
				classCount[c] = 0
			classCount[c] += 1
		sortecClassCount = sorted(classCount,key=operator.itemgetter(1),reverse=True)
		return sortecClassCount[0][0]

	def createTree(self,dataSet,labels):
		classList = [sample[-1] for sample in dataSet]
		if len(set(classList)) == 1:
			return classList[0]
		if len(dataSet[0])== 1:
			return majorityCnt(classList)
		bestFeature = self.chooseBestFeatureTosplit(dataSet)
		bestFeatureLabel = labels[bestFeature]
		tree = {bestFeatureLabel:{}}
		del(labels[bestFeature])
		featureVals = [sample[bestFeature] for sample in dataSet]
		uniqueVals = set(featureVals)
		for value in uniqueVals:
			sublabels = labels[:]
			subDataSet = self.splitDataSet(dataSet,bestFeature,value)
			tree[bestFeatureLabel][value] = self.createTree(subDataSet,sublabels)
		return tree


#test
dataSet = [[1,1,'y'],[1,1,'y'],[1,0,'n'],[0,1,'n'],[0,1,'n']]
labels = ['no facing','flippers']
dt = DecisionTree()
print(dt.shannonEnt(dataSet))
print(dt.splitDataSet(dataSet,0,1))
print(dt.chooseBestFeatureTosplit(dataSet))
print(dt.createTree(dataSet,labels))

