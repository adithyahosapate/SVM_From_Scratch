from svm import SVM
import numpy as np
import matplotlib.pyplot as plt
X1=np.random.multivariate_normal([1,2],[[1,0],[0,1]],size=100)
Y=np.ones(100)
X2=np.random.multivariate_normal([-10,5],[[1,0],[0,1]],size=100)
X=np.concatenate((X1,X2),axis=0)
Y=np.concatenate((Y,-Y))
svm=SVM(X,Y,"polynomial")
alphas=svm.optimize()
svm.plot()

#Test 2

points_array=np.array([
	[1,2],
	[1.2,3.5],
	[1,7],
	[1.7,5.6],
	[5,7],
	[3.0,4.7],
	[5,8],
	[-1,1],
	[1,7]

	])
labels_array=np.array([-1,1,1,-1,1,-1,1,1,1])
svm=SVM(points_array,labels_array,"gaussian")
alphas=svm.optimize()
svm.plot()