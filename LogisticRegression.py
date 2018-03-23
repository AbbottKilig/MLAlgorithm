# -*- coding:utf-8 -*-
import numpy as np

class LogisticRegression(object):
	"""docstring for LogisticRegression"""
	def __init__(self, iterations=500,epos=0.01):
		super(LogisticRegression, self).__init__()
		self.iterations = iterations
		self.epos = epos

	def sigmoid(self,inputX):
		return 1/(1 + np.exp(-inputX))

	def gradientDecrease(self,X,y,iterations,epos):
		n = X.shape[0]
		weight = np.zeros((n,1))
		for i in range(iterations):
			yhat = self.sigmoid(np.dot(weight.T,X))
			loss = np.sum((yhat - y)**2)
			dw = 2 * np.dot(X,np.dot(X.T,weight) - y.T)
			weight = weight - epos * dw
		return weight

	def stocGradientDecrease(self,X,y,iterations,epos):
		n,m = X.shape
		weight = np.zeros((n,1))
		for i in range(iterations):
			for j in range(m):
				dataIndex = np.random.random_integers(0,m-1)
				sampleX = X[:,dataIndex]
				sampleY = y[0][dataIndex]
				yhat = self.sigmoid(np.sum(np.dot(weight.T,sampleX)))
				loss = np.sum((yhat - y[:,dataIndex])**2)
				dw = 2 * sampleX *(np.dot(sampleX.T,weight) - sampleY)
				weight = weight - epos * dw
		return weight
	def fit(self,X,y,algorithm='gd'):
		'''
		params:
		X: n * m
		y: 1 * m

		'''
		if algorithm == 'sgd':
			self.weight == self.stocGradientDecrease(X,y,self.iterations,self.epos)
		else:
			self.weight = self.gradientDecrease(X,y,self.iterations,self.epos)

	def predict(self,X):
		return np.dot(self.weight.T,X)


#Test
W = np.array([[1],[0],[2]])
X = np.random.randn(3,5)
y = np.dot(W.T,X)

lr = LogisticRegression()
lr.fit(X, y)
print(lr.weight)

lr.fit(X, y, algorithm='sgd')
print(lr.weight)



