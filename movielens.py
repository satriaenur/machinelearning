import numpy as np
global allData


allData = np.genfromtxt('movielens/data/u.data',delimiter='\t', dtype='i8')
allItem = np.genfromtxt('movielens/data/u.item',delimiter='|', dtype=None)
def jaccardCoef(movie1, movie2):
	same1,same2 = [],[]
	for i in allData:
		if i[1] == movie1: same1.append(i[0])
		if i[1] == movie2: same2.append(i[0])
	same1,same2 = np.array(same1), np.array(same2)
	both = np.intersect1d(same1,same2).shape[0] + 0.0
	return both/(same1.shape[0] + same2.shape[0] - both)

def closerJaccard(movie,n):
	bestJaccardCoef = {i: [0,0] for i in range(n)}
	for i in allItem:
		if i[0] != movie:
			jacc = jaccardCoef(movie, i[0])
			for j in bestJaccardCoef:
				if bestJaccardCoef[j][1] < jacc:
					bestJaccardCoef[j] = [i[0],jacc]
					break
	return bestJaccardCoef

def corrRating(movie1, movie2):
	same1,same2 = [],[]
	rate1,rate2 = {}, {}
	for i in allData:
		if i[1] == movie1:
			same1.append(i[0])
			rate1[i[0]] = i[2]
		if i[1] == movie2:
			same2.append(i[0])
			rate2[i[0]] = i[2]
	same1,same2 = np.array(same1), np.array(same2)
	both = np.intersect1d(same1,same2)

	if both.shape[0] == 0:
		return 0

	bothrate1,bothrate2 = [],[]
	
	for i in both:
		bothrate1.append(rate1[i])
		bothrate2.append(rate2[i])
	avg1, avg2, top, down1, down2 = np.average(bothrate1), np.average(bothrate2),0.0, 0.0, 0.0
	for i in xrange(len(bothrate1)):
		top += (bothrate1[i] - avg1)*(bothrate2[i] - avg2)
		down1 += (bothrate1[i] -avg1)**2
		down2 += (bothrate2[i] - avg2)**2

	return top / (np.sqrt(down1) * np.sqrt(down2))

def closerCorrRating(movie,n):
	bestCorr = {i: [0,0] for i in range(n)}
	for i in allItem:
		if i[0] != movie:
			corr = corrRating(movie, i[0])
			for j in bestCorr:
				if bestCorr[j][1] < corr:
					bestCorr[j] = [i[0],corr]
					break
	return bestCorr
# print jaccardCoef(59,60)
# print closerJaccard(894, 5)
# print corrRating(59,60)
print closerCorrRating(894, 5)