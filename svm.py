import cvxopt
from cvxopt import matrix
import numpy as np
from kernels import Kernel
import matplotlib.pyplot as plt
import matplotlib.colors as colors
class SVM:
	def __init__(self,dataset,labels,kernel='linear'):
		self.dataset=dataset
		self.labels=labels
		
		if kernel=='linear':
			self.kernel=Kernel.linear()
		elif kernel=='gaussian':
			self.kernel=Kernel.gaussian()	
		elif kernel=='polynomial':
			self.kernel=Kernel.polynomial()	
		elif kernel=='hyperbolic_tangent':
			self.kernel=Kernel.hyperbolic_tangent()		
	def optimize(self):
		def create_gram_matrix():
			gram_matrix=np.zeros([self.dataset.shape[0],self.dataset.shape[0]])
			for i,data in enumerate(self.dataset):
				for (j,data_2) in enumerate(self.dataset):
					gram_matrix[i][j]=self.labels[i]*self.labels[j]*self.kernel(data,data_2)
			return gram_matrix
		N=self.dataset.shape[0]	
		self.gram_matrix=create_gram_matrix()
		P=matrix(self.gram_matrix)
		q=-matrix(np.ones([N,1]))
		G=matrix(-np.eye(N))
		h=matrix(np.zeros(N))
		A=matrix(np.reshape(self.labels,(1,-1)),(1,self.dataset.shape[0]),'d')
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
		support_vector=np.argmax(self.alphas)
		temp=0
		for alpha,label,data in zip(self.alphas,self.labels,self.dataset):
			temp+=label*alpha*self.kernel(self.dataset[support_vector],data)


		self.b=(self.labels[support_vector]-temp)[0]
		
		def evaluate(point):
			value=0

			for alpha,label,data in zip(self.alphas,self.labels,self.dataset):
				value+=label*alpha*self.kernel(point.T,data)

			return value.T+self.b
			

		mesh = np.meshgrid(np.linspace(np.min(self.dataset.T[0]), np.max(self.dataset.T[0]), 100),
				np.linspace(np.min(self.dataset.T[1]),np.max(self.dataset.T[1]), 100))
		xx,yy=mesh
		
		Z=[evaluate(np.array([xxx,yyy])) for xxx, yyy in zip(xx,yy)]
		
		fig=plt.figure(figsize=(5,5))
		cmap = colors.ListedColormap(['#FF0033','#99FFFF'])
		bounds = np.array([-1,0,1])
		
		pcm = plt.pcolormesh(xx, yy, Z,
                       		cmap=cmap,vmin=-1,vmax=1)
		plt.contour(xx, yy, Z, levels=[-1,0,1], linewidths=2,
                colors='k')

		colrs={1:'#000000',-1:'b'}
		for i,x in enumerate(self.dataset):
			plt.scatter(x[0],x[1],color=colrs[self.labels[i]])
		plt.show()

				
				




						

