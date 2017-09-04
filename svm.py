import cvxopt
import numpy as np
from Kernel import kernel

# TODO - optimize svms using Quadratic programming
class SVM:
	def __init__(self,dataset,labels,kernel):
		self.dataset=dataset
		self.labels=labels
		self.kernel=kernel

	def optimize(self):
		def create_gram_matrix(self):
			self.gram_matrix=np.zeroes(self.dataset.shape[0])
			for i,data in enumerate(dataset):
				for (i,data_2) in enumerate(dataset):
					gram_matrix[i][j]=kernel(data,data_2)
			return self.gram_matrix		