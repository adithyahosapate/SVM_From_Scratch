import numpy as np

class Kernel:
	#Source:wikipedia SVM kernel

	def linear():
		def f(x,y):
		#Linear kernel
			return np.inner(x,y)
		return f	

	def gaussian():
		def f(x,y,stddev=0.1):
			exponent= -np.sqrt(np.linalg.norm(x-y,axis=0)**2/(2*stddev**2))
			return np.exp(exponent)
		return f

	def polynomial():
		def f(x,y,offset=1,d=5):
			base=offset+np.inner(x,y)
			return np.power(base,d)
		return f
	def hyperbolic_tangent():			 	
		def f(x,y,kappa=1,c=1):
			return np.tanh(kappa+np.dot(x,y)+c)
		return f

