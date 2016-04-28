#!/usr/bin/python
import cv2,json
from skimage import feature
import glob,decimal,csv
import numpy as np
from matplotlib import pyplot as plt
from heapq import heapify, heappush, heappop
#queryimage=cv2.imread('queryimage.png')
#
#queryimage_features=f.cal_feature(queryimage)##

class LocalBinaryPatterns:
	def __init__(self, numPoints, radius):
		# store the number of points and radius
		self.numPoints = numPoints
		self.radius = radius
 
	def describe(self, image, eps=1e-7):
		# compute the Local Binary Pattern representation
		# of the image, and then use the LBP representation
		# to build the histogram of patterns
		lbp = feature.local_binary_pattern(image, self.numPoints,self.radius, method="uniform")
		(hist, _) = np.histogram(lbp.ravel(),
			bins=np.arange(0, self.numPoints + 2),
			range=(0, self.numPoints + 1))
 
		# normalize the histogram
		hist = hist.astype("float")
		hist /= (hist.sum() + eps)
 
		# return the histogram of Local Binary Patterns
		return hist
	def chi2_distance(self, histA, histB, eps = 1e-10):
		# compute the chi-squared distance
		d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
			for (a, b) in zip(histA, histB)])

		# return the chi-squared distance
		return d

t=LocalBinaryPatterns(12,8)

class Feature:
	def __init__(self, bins):
		self.bins = bins

	def cal_feature(self, image):
		image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
		features = []
		(l, b) = image.shape[:2]
		(x_centre, y_centre) = (int(l*0.5), int(b*0.5))
		parts = [(0, x_centre, 0, y_centre), (x_centre, b, 0, y_centre), (x_centre, b, y_centre, l),
			(0, x_centre, y_centre, l)]

		(axesX, axesY) = (int(b * 0.75) / 2, int(l * 0.75) / 2)
		ellipMask = np.zeros(image.shape[:2], dtype = "uint8")
		cv2.ellipse(ellipMask, (x_centre, y_centre), (axesX, axesY), 0, 0, 360, 255, -1)####

		for (startX, endX, startY, endY) in parts:
			mask = np.zeros(image.shape[:2], dtype = "uint8")
			cv2.rectangle(mask, (startX, startY), (endX, endY), 255, -1)
			mask = cv2.subtract(mask, ellipMask)
			hist = self.cal_hist(image, mask)
			features.extend(hist)

		hist = self.cal_hist(image, ellipMask)
		features.extend(hist)
		return features

	def cal_hist(self, image, mask):
         	hist = cv2.calcHist([image], [0, 1, 2], mask, self.bins,
			[0, 180, 0, 256, 0, 256])
		hist = cv2.normalize(hist).flatten()
		return hist


	def search(self, queryFeatures, limit = 10):
		# initialize our dictionary of results
		global results

		with open('index','rb') as f:
			
			reader = csv.reader(f)
			i=0
			for row in reader:
				
				if i%2==0:	
					features = [float(x) for x in row[1:]]
					d = self.chi2_distance(features, queryFeatures,0.3)
					if row[0] in results.keys():
						results[row[0]] += d
					else:
						results[row[0]]  = d
					i+=1
				elif i%2==1:
					texture = [float(x) for x in row[1:]]
					d = self.chi2_distance(texture, queryimage_texture,0.7)
					if row[0] in results.keys():
						results[row[0]] += d
					else:
						results[row[0]]  = d	
					i=0	
			
			#f.close()
		results = sorted([(v, k) for (k, v) in results.items()])

		#return results[:limit]

	def chi2_distance(self, histA, histB,weight,eps = 1e-10):
		# compute the chi-squared distance
		d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
			for (a, b) in zip(histA, histB)])

		# return the chi-squared distance
		return weight*d
results={}
f2=Feature((8,12,3))
#uncomment this to make index file
files=glob.glob('/home/user/Desktop/sem2/SMAI/project/Images/*')
#files=glob.glob('/home/tanushri/Pictures/anu/new year 2016/*')
index=open('index','a')
'''flag=0
for f in files:
	
	ID = f[f.rfind("/") + 1:]
	print ID
	if ID=='myntra_4936(2).jpg':
		flag=1
		continue
	if flag==0 and ID!='myntra_4936(2).jpg':
		continue
	
	img=cv2.imread(f,3)
	try:
		feature1=f2.cal_feature(img)
	except:
		continue
	for i in range(0,len(feature1)):
		feature1[i]=str(feature1[i])
        #feature.append(ID)
	#print feature
	#j=json.dumps(feature)
	index.write("%s,%s\n" % (ID, ",".join(feature1)))
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	texture=t.describe(gray)
	index.write(ID)
	for i in range(0,len(texture)):
		index.write(","+str(texture[i]))
		texture[i]=str(texture[i])
	index.write("\n")
	
index.close()'''
queryimage=cv2.imread('/home/user/Desktop/polo.jpeg',4)
queryimage_features=f2.cal_feature(queryimage)##
#f2.search(queryimage_features)
gray = cv2.cvtColor(queryimage, cv2.COLOR_BGR2GRAY)
queryimage_texture=t.describe(gray)
f2.search(queryimage_features)


'''def auto_canny(image, sigma=0.33):
	v = np.median(image)
 	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
	return edged

img=queryimage
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (3, 3), 0)
wide = cv2.Canny(blurred, 10, 200)
tight = cv2.Canny(blurred, 225, 250)
auto = auto_canny(blurred)
template= np.hstack([tight])
#template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
(tH, tW) = template.shape[:2]

alist=[]
found=None
for f in files:
	ID = f[f.rfind("/") + 1:]
	img=cv2.imread(f,3)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	found=None
	for scale in np.linspace(0.2, 1.0, 20)[::-1]:
		
		
		#r = gray.shape[1] / float(resized.shape[1])

		
		edged = cv2.Canny(gray, 50, 200)
		result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF)
		(_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
		if found==None:
			found=maxVal
			id2=ID
			heappush(alist,(found,ID))
		elif found<maxVal :
			if id2==ID:
				r=alist.index((found,id2))
				alist.pop(r)
				found=maxVal
				heappush(alist,(found,id2))
				heapify(alist)
			else:
				found=maxVal
				id2=ID
				heappush(alist,(found,id2))
		resized = cv2.resize(gray, (int(gray.shape[0] * scale),int(gray.shape[1] * scale)))
		gray=resized
		if resized.shape[0] < tH or resized.shape[1] < tW:
			break
		



while len(alist)>10:
	alist.pop(-1)

finalresult=[]
flag=0

for j in results:
	for k in alist:
		if k[1]==j[1]:
			finalresult.append(j[1])
		if len(finalresult)>10:
			flag=1
			break
	if flag==1:
		break





#
path='/home/tanushri/Pictures/anu/new year 2016/'
'''
results=results[:20]
path='/home/user/Desktop/sem2/SMAI/project/Images/'
for k in results:
	img = cv2.imread(path+k[1],4)
	print path+k[1]
	plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
	plt.show()


