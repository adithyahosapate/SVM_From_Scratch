import numpy as np
import matplotlib.pyplot as plt


class SVM:



   def __init__(self,dataset,labels):
      self.dataset=dataset
      self.labels=labels

   def classify(self,iterations,learning_rate,kernel):
      self.iterations=iterations
      self.kernel=kernel
      self.learning_rate=learning_rate
      self.w=np.random.normal(0,1,size=self.dataset.shape[1])
      self.b=0
      for iteration in range(self.iterations):
          lamb=1/(iteration+1)
          for i,x in enumerate(self.dataset):
              if self.labels[i]*(np.dot(x,self.w)+self.b)<1:
                  self.w=self.w+self.learning_rate*(self.labels[i]*x-2*lamb*self.w)
                  self.b=self.b+self.learning_rate*self.labels[i]
              else:
                  self.w=self.w+self.learning_rate*(-2*self.w*lamb)
      return self.w,self.b
   def plot(self):
       def evaluate_svm(weight,bias,point):
           return np.dot(weight,point)+bias
              #def plot_svm():
       xx, yy = np.meshgrid(np.linspace(np.min(self.dataset.T[0]), np.max(self.dataset.T[0]), 1000),
                         np.linspace(np.min(self.dataset.T[1]),np.max(self.dataset.T[1]), 1000))
       Z=[evaluate_svm(self.w,self.b,np.array([xxx,yyy])) for xxx,yyy in zip(xx,yy)]

       plt.figure(figsize=(5,5))


       plt.contour(xx, yy, Z, levels=[0], linewidths=2,
                colors='k')
       plt.pcolormesh(xx,yy,Z)

       colrs={1:'b',-1:'r'}
       for i,x in enumerate(self.dataset):
              plt.scatter(x[0],x[1],color=colrs[self.labels[i]])
       plt.show()
