import numpy as np
from numpy import random
import idx2numpy
import matplotlib.pyplot as plt

global oriimages, labels, images, LEN_DATA

LEN_DATA = 5000

oriimages = idx2numpy.convert_from_file('mnist/train-images-idx3-ubyte')[:LEN_DATA]
labels = idx2numpy.convert_from_file('mnist/train-labels-idx1-ubyte')[:LEN_DATA]
images = oriimages.reshape(oriimages.shape[0],28*28)

def dataVerification():
	f, x = plt.subplots(10, 10, sharex='col', sharey='row')
	for i in xrange(10):
		for j in xrange(10):
			idx = random.randint(0,LEN_DATA)
			print labels[idx],
			x[i][j].imshow(oriimages[idx],cmap="gray")
		print "\n"
	
	plt.show()

def meanModel(traindata,labeldata):
	avgdata,totaldata = [0.0 for i in xrange(10)],[0.0 for i in xrange(10)]
	for x in xrange(traindata.shape[0]):
		avgdata[labeldata[x]] += np.average(traindata[x])
		totaldata[labeldata[x]] += 1
	return (map(lambda x,y: x/y, avgdata,totaldata))

def euclidianMean(testdata,model):
	result = []
	for x in testdata:
		avgx, _min = np.average(x), []
		for y in xrange(10):
			score = np.sqrt((avgx - model[y])**2)
			_min = [y,score] if (_min == [] or score < _min[1]) else _min
		result.append(_min[0])
	return result

train, labeltrain = images[:2500], labels[:2500]
testing, labeltest = images[2500:], labels[2500:]

print euclidianMean(testing, meanModel(train,labeltrain))
print labeltest