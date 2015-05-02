import numpy as np
import numpy.linalg as la
from numpy.linalg import pinv
from numpy.linalg import inv
import cv2, math
from os import listdir
from tempfile import TemporaryFile
from datetime import datetime, timedelta

#np.set_printoptions(threshold=np.nan)
dist_dir = 'dist/'
rotate = 5 # inversely proportional to number of rows
iter = 12 # inversely proportional to number of columns
n = 20
p = 31

# returns true if p is to the left of the line (p1, p2)
def isLeft(p, p1, p2):
	return ((p2[0] - p1[0])*(p[1] - p1[1]) - (p2[1] - p1[1])*(p[0] - p1[0])) > 0

# calculates the iterative u step
def fu(v):
	u_star = np.zeros((n, n), dtype=float)
	for k in range(0, r):
		u_star = np.add(u_star, np.subtract(matrices[k], mean).dot(pinv(v)).dot(np.transpose(np.subtract(matrices[k], mean))))
	u_star *= (1/float(p*r))
	return u_star

# calculates the iterative v step
def fv(u):
	v_plus = np.zeros((p, p), dtype=float)
	for k in range(0, r):
		v_plus = np.add(v_plus, np.transpose(np.subtract(matrices[k], mean)).dot(pinv(u)).dot(np.subtract(matrices[k], mean)))
	v_plus *= (1/float(n*r))
	return v_plus

# takes in a filename, returns the euler characteristic cdf matrix
def getMatrix(filename):
	im = cv2.imread('png/'+filename)
	imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
	ret,thresh = cv2.threshold(imgray,127,255,0)
	_, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	c = contours[0]
	v = np.squeeze(c)
	#v = v[1::2]

	origin = (128, 128)

	m = []

	for i in range(36, 3600, 36*rotate):
		angle = i/10.0
		vec = []
		points = []
		for step in range(181, 0, -iter): #362 is maximum number of pixels we can step
			points.append((-np.cos(angle)*step+origin[0], -np.sin(angle)*step+origin[1]))
		for step in range(1, 181, iter): #362 is maximum number of pixels we can step
			points.append((np.cos(angle)*step+origin[0], np.sin(angle)*step+origin[1]))
		for point in points:
			slope = -float(point[0]-origin[0])/float(point[1]-origin[1])
			point2 = (point[0]+10, point[1]+(slope*10))
			num_v = 0
			num_e = 0
			for p in range(0, len(v)):
				if isLeft(v[p], point, point2):
					num_v += 1
				if p < len(v)-1:
					if isLeft(v[p], point, point2) > 0 and isLeft(v[p+1], point, point2):
						num_e += 1
				else:
					if isLeft(v[p], point, point2) > 0 and isLeft(v[0], point, point2):
						num_e += 1
			ec = num_v - num_e
			if len(vec) == 0:
				vec.append(ec)
			else:
				vec.append(ec+vec[len(vec)-1])
		m.append(vec)
	return m

if __name__ == "__main__":
	dir = listdir('png/')
	if '.DS_Store' in dir:
		dir.remove('.DS_Store')
	names = []
	for filename in dir:
		name = filename.split('-')[0]
		if name not in names:
			names.append(name)
	names = sorted(names)
	for name in names: # for each shape
		print name
		matrices = []
		max = 19
		for x in range(1, max):
			matrices.append(getMatrix(name+'-'+str(x)+'.png'))

		# computes the mean of each matrix
		#norm contains a matrix for each image

		mean = np.zeros((len(matrices[0]), len(matrices[0][0])))
		for m in matrices: # compute the mean
			m = np.array(m)
			for j in range(0, len(m)): # for each row
				mean[j] = np.add(mean[j], m[j]) # cumulative sum
		for k in range(0, len(mean)): # for each row in mean
			mean[k] = np.array(mean[k])/len(matrices) # set as mean


		#computes the mle parameters for matrix normal
		e_1 = 1.0e-10 # small values
		e_2 = 1.0e-10
		n = len(matrices[0])
		p = len(matrices[0][0])
		r = len(matrices)
		v_star = np.identity(p, dtype=float) #v_zero
		u_star = fu(v_star)
		v_plus = fv(u_star)
		u_plus = fu(v_plus)
		time = datetime.now()
		period = timedelta(seconds=5) # a manual override for when v gets caught up too long
		while la.norm(np.subtract(u_plus, u_star), 2) > e_1 or la.norm(np.subtract(v_plus, v_star), 2) > e_2:
			if datetime.now()>time+period:
				break
			if la.norm(np.subtract(u_plus, u_star), 2) > e_1:
				u_star = u_plus
			v_star = v_plus
			v_plus = fv(u_star)
			u_plus = fu(v_plus)

		m = mean
		u = u_star
		v = v_star

		dist = np.array([m, u, v])
		#print dist
		outfile = dist_dir+name
		np.save(outfile, dist)
		print name+" saved"
