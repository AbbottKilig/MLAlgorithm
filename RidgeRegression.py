import numpy as np

class RidgeRegression(object):
	"""RidgeRegression"""
	def fit(self,X,y,lambd=0.1):
		'''
		X dimentions: n * m
		y dimentions: 1 * m
		weight dimentions: n * 1
		'''
		xxT = np.dot(X,X.T)
		mat = xxT - np.eye(xxT.shape[0]) * lambd
		if np.linalg.det(mat) == 0:
			raise "lambda connot be zero"
		self.weight = np.dot(np.dot(np.linalg.inv(mat),X),y.T)
	def predcit(self,X):
		'''
		predict values of X
		'''
		return self.weight.T * X
		
#test
rg = RidgeRegression()
X = np.random.randn(3,3)
w = np.array([[1],[1],[1]])
y = np.dot(w.T,X)
print(X.shape)
print(w.shape)
print(y.shape)

rg.fit(X,y,0)
print(rg.weight)