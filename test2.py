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
svm=SVM(points_array,labels_array,"polynomial")
alphas=svm.optimize()
svm.plot()
