import json
import os, sys
import Image

f=open('centroids2.txt','r')
centroids=[]
large=[]
for i in range(0,12):
	s=f.readline()
	centroids.append(json.loads(s))
	

f.close()

from scipy import signal
import glob
import csv,math,random,cv2,json
from collections import defaultdict
from skimage import feature
import glob,decimal,csv
import numpy as np
from matplotlib import pyplot as plt
from heapq import heapify, heappush, heappop
# sift features
Nangles = 8
Nbins = 4
Nsamples = Nbins**2
alpha = 9.0
angles = np.array(range(Nangles))*2.0*np.pi/Nangles


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
         	hist = cv2.calcHist([image], [0, 1, 2], mask, self.bins,[0, 180, 0, 256, 0, 256])
		hist = cv2.normalize(hist,10.0).flatten()
		return hist


	def search(self, queryFeatures, limit = 10):
		# initialize our dictionary of results
		global results

		with open('index.txt','rb') as f:
			
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

	def chi2_distance(self, histA, histB,weight,eps = 1e-10):
		# compute the chi-squared distance
		d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
			for (a, b) in zip(histA, histB)])

		# return the chi-squared distance
		return weight*d

	

def gen_dgauss(sigma):
    '''
    generating a derivative of Gauss filter on both the X and Y
    direction.
    '''
    fwid = np.int(2*np.ceil(sigma))
    G = np.array(range(-fwid,fwid+1))**2
    G = G.reshape((G.size,1)) + G
    G = np.exp(- G / 2.0 / sigma / sigma)
    G /= np.sum(G)
    GH,GW = np.gradient(G)
    GH *= 2.0/np.sum(np.abs(GH))
    GW *= 2.0/np.sum(np.abs(GW))
    return GH,GW

class DsiftExtractor:
    '''
    The class that does dense sift feature extractor.
    Sample Usage:
        extractor = DsiftExtractor(gridSpacing,patchSize,[optional params])
        feaArr,positions = extractor.process_image(Image)
    '''
    def __init__(self, gridSpacing, patchSize,
                 nrml_thres = 1.0,\
                 sigma_edge = 0.8,\
                 sift_thres = 0.2):
        '''
        gridSpacing: the spacing for sampling dense descriptors
        patchSize: the size for each sift patch
        nrml_thres: low contrast normalization threshold
        sigma_edge: the standard deviation for the gaussian smoothing
            before computing the gradient
        sift_thres: sift thresholding (0.2 works well based on
            Lowe's SIFT paper)
        '''
        self.gS = gridSpacing
        self.pS = patchSize
        self.nrml_thres = nrml_thres
        self.sigma = sigma_edge
        self.sift_thres = sift_thres
        # compute the weight contribution map
        sample_res = self.pS / np.double(Nbins)
        sample_p = np.array(range(self.pS))
        sample_ph, sample_pw = np.meshgrid(sample_p,sample_p)
        sample_ph.resize(sample_ph.size)
        sample_pw.resize(sample_pw.size)
        bincenter = np.array(range(1,Nbins*2,2)) / 2.0 / Nbins * self.pS - 0.5 
        bincenter_h, bincenter_w = np.meshgrid(bincenter,bincenter)
        bincenter_h.resize((bincenter_h.size,1))
        bincenter_w.resize((bincenter_w.size,1))
        dist_ph = abs(sample_ph - bincenter_h)
        dist_pw = abs(sample_pw - bincenter_w)
        weights_h = dist_ph / sample_res
        weights_w = dist_pw / sample_res
        weights_h = (1-weights_h) * (weights_h <= 1)
        weights_w = (1-weights_w) * (weights_w <= 1)
        # weights is the contribution of each pixel to the corresponding bin center
        self.weights = weights_h * weights_w
        #pyplot.imshow(self.weights)
        #pyplot.show()
        
    def process_image(self, image, positionNormalize = True,\
                       verbose = True):
        '''
        processes a single image, return the locations
        and the values of detected SIFT features.
        image: a M*N image which is a numpy 2D array. If you 
            pass a color image, it will automatically be converted
            to a grayscale image.
        positionNormalize: whether to normalize the positions
            to [0,1]. If False, the pixel-based positions of the
            top-right position of the patches is returned.
        
        Return values:
        feaArr: the feature array, each row is a feature
        positions: the positions of the features
        '''

        image = image.astype(np.double)
        if image.ndim == 3:
            # we do not deal with color images.
            image = np.mean(image,axis=2)
        # compute the grids
        H,W = image.shape
        gS = self.gS
        pS = self.pS
        remH = np.mod(H-pS, gS)
        remW = np.mod(W-pS, gS)
        offsetH = remH/2
        offsetW = remW/2
        gridH,gridW = np.meshgrid(range(offsetH,H-pS+1,gS), range(offsetW,W-pS+1,gS))
        gridH = gridH.flatten()
        gridW = gridW.flatten()
        if verbose:
            print 'Image: w {}, h {}, gs {}, ps {}, nFea {}'.\
                    format(W,H,gS,pS,gridH.size)
        feaArr = self.calculate_sift_grid(image,gridH,gridW)
        feaArr = self.normalize_sift(feaArr)
        if positionNormalize:
            positions = np.vstack((gridH / np.double(H), gridW / np.double(W)))
        else:
            positions = np.vstack((gridH, gridW))
        return feaArr, positions

    def calculate_sift_grid(self,image,gridH,gridW):
        '''
        This function calculates the unnormalized sift features
        It is called by process_image().
        '''
        H,W = image.shape
        Npatches = gridH.size
        feaArr = np.zeros((Npatches,Nsamples*Nangles))

        # calculate gradient
        GH,GW = gen_dgauss(self.sigma)
        IH = signal.convolve2d(image,GH,mode='same')
        IW = signal.convolve2d(image,GW,mode='same')
        Imag = np.sqrt(IH**2+IW**2)
        Itheta = np.arctan2(IH,IW)
        Iorient = np.zeros((Nangles,H,W))
        for i in range(Nangles):
            Iorient[i] = Imag * np.maximum(np.cos(Itheta - angles[i])**alpha,0)
            #pyplot.imshow(Iorient[i])
            #pyplot.show()
        for i in range(Npatches):
            currFeature = np.zeros((Nangles,Nsamples))
            for j in range(Nangles):
                currFeature[j] = np.dot(self.weights,\
                        Iorient[j,gridH[i]:gridH[i]+self.pS, gridW[i]:gridW[i]+self.pS].flatten())
            feaArr[i] = currFeature.flatten()
        return feaArr

    def normalize_sift(self,feaArr):
        '''
        This function does sift feature normalization
        following David Lowe's definition (normalize length ->
        thresholding at 0.2 -> renormalize length)
        '''
        siftlen = np.sqrt(np.sum(feaArr**2,axis=1))
        hcontrast = (siftlen >= self.nrml_thres)
        siftlen[siftlen < self.nrml_thres] = self.nrml_thres
        # normalize with contrast thresholding
        feaArr /= siftlen.reshape((siftlen.size,1))
        # suppress large gradients
        feaArr[feaArr>self.sift_thres] = self.sift_thres
        # renormalize high-contrast ones
        feaArr[hcontrast] /= np.sqrt(np.sum(feaArr[hcontrast]**2,axis=1)).\
                reshape((feaArr[hcontrast].shape[0],1))
        return feaArr

class SingleSiftExtractor(DsiftExtractor):
    '''
    The simple wrapper class that does feature extraction, treating
    the whole image as a local image patch.
    '''
    def __init__(self, patchSize,
                 nrml_thres = 1.0,\
                 sigma_edge = 0.8,\
                 sift_thres = 0.2):
        # simply call the super class __init__ with a large gridSpace
        DsiftExtractor.__init__(self, patchSize, patchSize, nrml_thres, sigma_edge, sift_thres)   
    
    def process_image(self, image):
        return DsiftExtractor.process_image(self, image, False, False)[0]


def getdist(a, b):
	    s=0.0
	    #print len(b),len(a)
	    
	    for i1 in range(0,len(b)):
		s=s+(a[i1]-b[i1])*(a[i1]-b[i1])
	
	 
	    return math.sqrt(s)


def kmeans(points):

	global l
	
	for i in range(0,len(points)):
		smallestdist=getdist(points[i], centroids[i][0])
		index=0
		for j in range(1,numofclusters):
			distance = getdist(points[i], centroids[i][j])
			if distance < smallestdist:
		            		smallestdist = distance
		            		index = j
		l.append(index)
	
	
from scipy import misc
t=LocalBinaryPatterns(12,8)
results={}
f2=Feature((8,12,3))
queryfeature=[]
extractor = SingleSiftExtractor(360)
numofclusters=100
path='/home/tanushri/Documents/SMAI/project/Images/'
l=[]
def matchsift(queryimage):
	global queryfeature,flag,results2
	
	image = np.mean(np.double(queryimage),axis=2)
	for i in range(0,4):
		for j in range(0,3):
		
			feaArrSingle = extractor.process_image(image[i*360:i*360+360,j*360:j*360+360])

			try:
				min=np.amin(feaArrSingle)
				max=np.amax(feaArrSingle)
			except:
				
				flag=1
				break
				
			
			tmp=[]
			for k in range(0,128):

				tmp.append(float(feaArrSingle[0][k]-min)/(max-min))
			queryfeature.append(tmp)
	kmeans(queryfeature)
	global results2
	f=open('final.txt','r')
	s='a'
	while len(s)>0:
		s=f.readline()
		if len(s)>0:
			l2=s.split(':')
			count=0
			l3=json.loads(l2[1])
			for i in range(0,len(l3)):
				if l[i]==l3[i]:
					count+=1
			results2[l2[0]]=count
	results2 = sorted([(v, k) for (k, v) in results2.items()])
	f.close()
name='/home/tanushri/Documents/SMAI/project/'+sys.argv[1]
queryimage=cv2.imread(name,4)
results2={}
queryfeature=[]
final=[]
flag=0
count=1
if queryimage.shape==(1440,1080):
	matchsift(queryimage)
else:
	 flag=1
if flag==1:
	count=3
	results2={}
	img = cv2.imread(name)
	img = cv2.resize(img, (1080, 1440)) 
	img = cv2.blur(img,(2,2))
	gray_seg = cv2.Canny(img, 0, 50)
	for i in range(0,img.shape[0]):
		for j in range(0,img.shape[1]):
	
			if gray_seg[i][j]==0:
				img[i][j][0]=255
				img[i][j][1]=255
				img[i][j][2]=255
			else:
				break
		for j in range(img.shape[1]-1,-1,-1):
			if gray_seg[i][j]==0:
				img[i][j][0]=255
				img[i][j][1]=255
				img[i][j][2]=255
			else:
				break
	cv2.imwrite("queryimage.jpg", img)
	queryimage=cv2.imread('/home/tanushri/Documents/SMAI/project/queryimage.jpg',4)
	gray = cv2.cvtColor(queryimage, cv2.COLOR_BGR2GRAY)
	queryimage_features=f2.cal_feature(queryimage)
	queryimage_texture=t.describe(gray)
	f2.search(queryimage_features)
	queryfeature=[]
	matchsift(queryimage)
	
else:
	print 'hi'
	for i in range(len(results2)-1,len(results2)-26,-1):
		print results2[i]
		if results2[i][0]>2:
			final.append(results2[i][1])	
	if len(final)<15:
		gray = cv2.cvtColor(queryimage, cv2.COLOR_BGR2GRAY)
		queryimage_features=f2.cal_feature(queryimage)
		queryimage_texture=t.describe(gray)
		f2.search(queryimage_features)

path='/home/tanushri/Documents/SMAI/project/Images/'

	
if len(results2)>0 and len(final)==0:	
	
	for i in range(len(results2)-1,len(results2)-26,-1):
		print results2[i]
		if results2[i][0]>count:
			final.append(results2[i][1])
if len(final)<15:
	for i in range(0,15-len(final)):
		if results[i][1] not in final:
			final.append(results[i][1])
	
for k in set(final):
	img = cv2.imread(path+k,4)
	print path+k
	plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
	plt.show()
