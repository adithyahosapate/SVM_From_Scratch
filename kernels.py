import numpy as np

class Kernel:
	#Source:wikipedia SVM kernels
	def __init__(self):
		pass

	def linear(self,x,y):
		#Linear kernel
		return np.inner(x,y)

	def gaussian(self,x,y,stddev):
		exponent= -np.sqrt(np.linalg.norm(x-y)**2/(2*stddev**2))
		return np.exp(exponent)

	def polynomial(self,x,y,offset,d):
		base=offset+np.inner(x,y)
		return np.pow(base,d)
				 	
	def hyperbolic_tangent(self,x,y,kappa,c):
		return np.tanh(kappa+np.dot(x,y)+c)


