import cvxopt
from cvxopt import matrix
import numpy as np
from kernels import Kernel
import matplotlib.pyplot as plt
import matplotlib.colors as colors
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
		P=matrix(self.gram_matrix)
		q=matrix(np.ones([self.dataset.shape[0],1]))
		G=matrix(-np.eye(self.dataset.shape[0]))
		h=matrix(np.zeros([self.dataset.shape[0]]))
		A=matrix(np.reshape(self.labels,(1,-1)),(1,200),'d')
		b=matrix(np.zeros(1))			
		solution=cvxopt.solvers.qp(P=P,
						q=q,
						G=G,
						h=h,
						A=A,
						b=b,
						solver=None)
		alphas=np.array(solution['x'])
		self.alphas=alphas
		return alphas
	def plot(self):
		support_vector=self.dataset[np.argmax(self.alphas)]
		temp=0
		for alpha,label,data in zip(self.alphas,self.labels,self.dataset):
			temp+=label*alpha*np.inner(data,support_vector)
		self.b=self.labels[np.argmax(self.alphas)]-temp
		def evaluate(point):
			value=0

			for alpha,label,data in zip(self.alphas,self.labels,self.dataset):
				value+=label*alpha*np.dot(data.T,point)

			return value.T+self.b
			

		mesh = np.meshgrid(np.linspace(np.min(self.dataset.T[0]), np.max(self.dataset.T[0]), 100),
				np.linspace(np.min(self.dataset.T[1]),np.max(self.dataset.T[1]), 100))
		xx,yy=mesh
		#print(mesh)
		#print(xx)
		Z=[evaluate(np.array([xxx,yyy])) for xxx, yyy in zip(xx,yy)]
		fig=plt.figure(figsize=(5,5))
		bounds = np.array([-1,0,1])
		norm = colors.BoundaryNorm(boundaries=bounds, ncolors=3)
		pcm = plt.pcolormesh(xx, yy, Z,
                       	norm=norm,
                       		cmap='RdBu_r')
		plt.contour(xx, yy, Z, levels=[-1,0,1], linewidths=2,
                colors='k')

		colrs={1:'b',-1:'r'}
		for i,x in enumerate(self.dataset):
			plt.scatter(x[0],x[1],color=colrs[self.labels[i]])
		plt.show()

				
				




						

