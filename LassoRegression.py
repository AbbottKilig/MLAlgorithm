#-*-coding:utf-8-*-
import numpy as np

class LassoRegression(object):
	"""docstring for LassoRegression"""
	def __init__(self, iters=300,epos=0.01):
		self.iters = iters
		self.epos = epos

	@staticmethod
	def  squareError(y,yhat):
		return np.sum((y - yhat) ** 2)

	def fit(self,X,y):
		'''
		params:
		X: n * m
		y: 1 * m
		'''

		n  = X.shape[0]
		self.weight = np.zeros((n,1))
		yhat = np.dot(self.weight.T,X)
		self.squareerror = self.squareError(y,yhat)
		for i in range(self.iters):
			for j in range(n):
				for k in [-1,1]:
					w = self.weight.copy()
					w[j] += self.epos * k
					yhat = np.dot(w.T,X)
					SETest = self.squareError(y,yhat)
					if SETest < self.squareerror:
					 	self.weight = w
					 	self.squareerror = SETest

#Test
W = np.array([[1],[0],[2]])
X = np.random.randn(3,5)
y = np.dot(W.T,X)

lr = LassoRegression()
lr.fit(X, y)
print(lr.weight)
