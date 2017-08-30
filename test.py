from svm import SVM
import numpy as np
import matplotlib.pyplot as plt
X1=np.random.multivariate_normal([1,2],[[1,0],[0,1]],size=100)
Y=np.random.choice([1],100)
X2=np.random.multivariate_normal([-10,5],[[1,0],[0,1]],size=100)
X=np.concatenate((X1,X2),axis=0)
Y=np.concatenate((Y,-Y))
svm=SVM(X,Y)
w,b=svm.classify(1000,0.01,"linear")
print(w,b)
svm.plot()
