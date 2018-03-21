# -*- coding:utf-8 -*-
import numpy as np

class OLS(object):
	"""training for OLS"""

		

	def fit(self,X,y):
		'''
		X dimentions: n * m
		y dimentions: 1 * m
		weight dimentions: n * 1
		'''
		mat = np.dot(X,X.T)
		if np.linalg.det(mat) == 0.0:
			raise "matrix connot do inverse"
		self.weight = np.dot(np.dot(np.linalg.inv(mat),X),y.T)
		print(self.weight.shape)
	def predict(self,X):
		'''
		predict values of X
		'''
		return self.weight.T * X


#test		
ols = OLS()
X = np.random.randn(3,3)
w = np.array([[1],[1],[1]])
y = np.dot(w.T,X)
print(X.shape)
print(w.shape)
print(y.shape)

ols.fit(X,y)
print(ols.weight)