import cvxopt
import numpy as np
from kernels import Kernel

# TODO - optimize svms using Quadratic programming
class SVM:
	def __init__(self,dataset,labels,kernel):
		self.dataset=dataset
		self.labels=labels
		self.kernel=kernel

	def optimize(self):
		def create_gram_matrix():
			gram_matrix=np.zeros([self.dataset.shape[0],self.dataset.shape[0]])
			for i,data in enumerate(self.dataset):
				for (j,data_2) in enumerate(self.dataset):
					gram_matrix[i][j]=self.labels[i]*self.labels[j]*np.inner(data,data_2)
			return gram_matrix
		self.gram_matrix=create_gram_matrix()			
		print(cvxopt.solvers.qp(P=-self.gram_matrix,
						q=np.ones([self.dataset.shape[0],1]),
						G=-np.eye(self.dataset.shape[0]),
						h=np.zeros([self.dataset.shape[0],1]),
						A=np.diag(self.labels),
						b=np.zeros([self.dataset.shape[0],1]),
						solver=None))

