import numpy as np
from numpy import random
import idx2numpy
import matplotlib.pyplot as plt

global oriimages, labels, images, LEN_DATA

LEN_DATA = 5000
TEST_TOTAL = 2500

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
	avgdata,totaldata = np.array([np.array([0.0 for i in xrange(traindata.shape[1])]) for i in xrange(10)]),[0.0 for i in xrange(10)]
	for x in xrange(traindata.shape[0]):
		avgdata[labeldata[x]] += traindata[x]
		totaldata[labeldata[x]] += 1
	return avgdata / np.array(totaldata).reshape(10,1)

def euclidianMean(testdata, label, model):
	result = []
	confmatrix = np.array([np.array([0.0 for i in xrange(4)]) for i in xrange(10)])
	for x in xrange(testdata.shape[0]):
		_min = []
		for y in xrange(10):
			score = np.sqrt(np.sum((testdata[x] - model[y])**2))
			_min = [y,score] if (_min == [] or score < _min[1]) else _min
		confmatrix += [0, 0, 0, 1]
		if _min[0] == label[x]:
			confmatrix[label[x]] += [1, 0, 0, -1]
		else:
			confmatrix[label[x]] += [0, 1, 0, -1]
			confmatrix[_min[0]] += [0, 0, 1, -1]
		result.append(_min[0])
	return np.array(result), confmatrix

def knn(traindata,labeltrain,testdata,labeltest, k):
	result = []
	confmatrix = np.array([np.array([0 for i in xrange(4)]) for i in xrange(10)])
	for x in xrange(testdata.shape[0]):
		coba = []
		for y in xrange(traindata.shape[0]):
			score = np.sqrt(np.sum((testdata[x] - traindata[y])**2))
			coba.append(score)
			if y == 0:
				_min = [[labeltrain[y],score] for xx in xrange(k)]
			else:
				for z in xrange(k):
					if _min[z][1] > score:
						_min = _min[:z]+[[labeltrain[y],score]]+_min[z:-1]
						break
		xresult = np.argmax(np.bincount(np.array(_min)[:,0].astype(int)))
		confmatrix += [0, 0, 0, 1]
		if xresult == labeltest[x]:
			confmatrix[labeltest[x]] += [1, 0, 0, -1]
		else:
			confmatrix[labeltest[x]] += [0, 1, 0, -1]
			confmatrix[xresult] += [0, 0, 1, -1]
		result.append(xresult)

	return np.array(result), confmatrix


train, labeltrain = images[:LEN_DATA - TEST_TOTAL].astype('int64'), labels[:LEN_DATA - TEST_TOTAL]
testing, labeltest = images[TEST_TOTAL:].astype('int64'), labels[TEST_TOTAL:]

print "Creating mean prototype...\n"
model1 = meanModel(train,labeltrain)
print "Collection euclidianMean model result...\n"
model1result, model1evaluate = euclidianMean(testing, labeltest, model1)
print "Collection 3-nn model result...\n"
model2result, model2evaluate = knn(train,labeltrain,testing, labeltest,3)
print "Euclidian Result:",model1result
print "KNN Result:",model2result
print "Real Label:",labeltest

print "Euclidian Conf.M\n",model1evaluate
print "Accuracy: ",np.average((model1evaluate[:,0] + model1evaluate[:,3] + 0.0)/TEST_TOTAL)
print "KNN Conf.M\n",model2evaluate
print "Accuracy: ",np.average((model2evaluate[:,0] + model2evaluate[:,3] + 0.0)/TEST_TOTAL)