import numpy as np
import numpy.linalg as la
from numpy.linalg import pinv
from numpy.linalg import inv
import cv2, math
from tempfile import TemporaryFile
import os
import random
import dist
#np.set_printoptions(threshold=np.nan)

dist_dir = dist.dist_dir
rotate = dist.rotate
iter = dist.iter
n = dist.n
p = dist.p

# returns most likely distribution given test matrix x
def predict(x):
	maximum = "null"
	for param in params: # iterate through all distributions
		(m, u, v, name) = param
		mtx = pinv(v)*(np.transpose(np.subtract(x, m)))*(pinv(u))*(np.subtract(x, m))
		trace = np.trace(mtx) #butterfy has a small trace
		l = 1.0e300*np.exp(-0.5*trace)/((la.norm(v)**(n/2.0))*(la.norm(u)**(p/2.0))) # likelihood, excluding the "2pi" term and multiplying by a large positive number (we get overflow otherwise)
		if maximum == "null":
			maximum = (l, name)
		elif l > maximum[0]:
			maximum = (l, name)
			print maximum
	return maximum

dir = os.listdir(dist_dir)
if '.DS_Store' in dir:
	dir.remove('.DS_Store')
params = []
for distribution in dir:
	param = np.load(dist_dir+distribution)
	param = np.append(param, distribution.split('.')[0]) #append dist name
	params.append(param)

# get test matrices (the 19th and 20th image for each shape)
tests = []
for distribution in dir:
	name = distribution.split('.')[0]
	tests.append(name+'-19.png') # test the 19th entry
	tests.append(name+'-20.png') # test the 20th entry

right = wrong = 0.0
for test in tests: # all test shapes
	name = test.split('-')[0]
	x = dist.getMatrix(test) # get matrix of test
	prediction = predict(x)
	if prediction[1] == name:
		right += 1
	else:
		wrong += 1
	print "Guess: "+str(prediction[1])+", Actual: "+name
print "Percent right: "+str(right/(right+wrong))

