import cvxopt
from cvxopt import matrix
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
		P=-self.gram_matrix
		q=np.ones([self.dataset.shape[0],1])
		G=-np.eye(self.dataset.shape[0])
		h=np.zeros([self.dataset.shape[0],1])
		A=matrix(np.reshape([self.labels],(1,200)),(1,200), 'd')
		b=np.zeros([self.dataset.shape[0],1])
		print(matrix(A))			
		print(cvxopt.solvers.qp(P=matrix(-self.gram_matrix),
						q=matrix(np.ones([self.dataset.shape[0],1])),
						G=matrix(-np.eye(self.dataset.shape[0])),
						h=matrix(np.zeros([self.dataset.shape[0],1])),
						A=A ,#matrix((self.labels).T),
						b=matrix(np.zeros([1,])),
						solver=None))
	
						

