'''
dsift.py: this function implements some basic functions that 
does dense sift feature extraction.
The descriptors are defined in a similar way to the one used in
Svetlana Lazebnik's Matlab implementation, which could be found
at:
http://www.cs.unc.edu/~lazebnik/
Yangqing Jia, jiayq@eecs.berkeley.edu
'''


from scipy import signal
import glob,time
import csv,math,random,cv2,Image
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
# sift features
Nangles = 8
Nbins = 4
Nsamples = Nbins**2
alpha = 9.0
angles = np.array(range(Nangles))*2.0*np.pi/Nangles

def pca(x):


	#queryimage=cv2.imread('/home/tanushri/Documents/SMAI/project/myntra_277.jpg',4)

	#x = cv2.cvtColor(queryimage, cv2.COLOR_BGR2GRAY)
	mean_vec = np.mean(x, axis=0)
	cov_mat = (x - mean_vec).T.dot((x - mean_vec)) / (x.shape[0]-1)
	cov_mat = np.cov(x.T)
	eig_vals, eig_vecs = np.linalg.eig(cov_mat)
	eig=[]
	for i in range(0,len(eig_vals)):
		x=eig_vals[i].real
		eig.append(x)
	eig=np.asarray(eig)
	eig_v=[]
	tmp=[]
	for i in range(0,1080):
		for j in range(0,1080):
			tmp.append(eig_vecs[i][j].real)
		eig_v.append(tmp)
		tmp=[]
	eig_v=np.asarray(eig_v)
	maxx=-9999
	for i in range(0,1080):
		if maxx<eig[i]:
			maxx=eig[i]
			index=i
	maxx2=-9999
	for i in range(0,1080):
		if maxx2<eig[i] and eig[i]<maxx:
			maxx2=eig[i]
			index2=i

	##pca 1 vs 2
	matrix_12 = np.hstack((eig_v[index].reshape(1080,1),
		              eig_v[index2].reshape(1080,1)))

	#Y12 = x.dot(matrix_12)
	Y12=np.dot(x,matrix_12)

	'''plt.figure(figsize=(6, 4))

	for i in range(0,Y12.shape[0]):
	
			plt.scatter(Y12[i, 0],
				    Y12[i, 1],
		
				    )
	plt.xlabel('Principal Component 1')
	plt.ylabel('Principal Component 2')
	plt.legend(loc='lower center')
	plt.tight_layout()
	plt.show()'''

	

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
    
if __name__ == '__main__':
    # ignore this. I only use this for testing purpose...
    from scipy import misc
    extractor = DsiftExtractor(8,16,1)
    flag=0
    #feaArr,positions = extractor.process_image(image)

    #pyplot.hist(feaArr.flatten(),bins=100)
    #pyplot.imshow(feaArr[:256])
    #pyplot.plot(np.sum(feaArr,axis=0))
   # pyplot.imshow(feaArr[np.random.permutation(feaArr.shape[0])[:256]])
    
    # test single sift extractor
    extractor = SingleSiftExtractor(360)
    files=glob.glob('/home/tanushri/Documents/SMAI/project/Images/*')
    #index=open('/media/tanushri/FE71-2EA4/sift3.txt','a')
    index=open('/home/tanushri/Documents/SMAI/project/sift2.txt','a')
    path='/home/tanushri/Documents/SMAI/project/Images/'
    count=0
    for f in files:
		try:
		    ID = f[f.rfind("/") + 1:]
		    
		    print ID,count
		    count+=1
		    image = cv2.imread(path+ID)
		    img = cv2.blur(image,(2,2))
		    gray_seg = cv2.Canny(img, 0, 50)
		    for i in range(0,img.shape[0]):
			for j in range(0,img.shape[1]):
	
					if gray_seg[i][j]==0:
						img[i][j][0]=0+255
						img[i][j][1]=0+255
						img[i][j][2]=0+255
					else:
						break
			for j in range(img.shape[1]-1,-1,-1):
					if gray_seg[i][j]==0:
						img[i][j][0]=0+255
						img[i][j][1]=0+255
						img[i][j][2]=0+255
					else:
						break

		    cv2.imwrite("queryimage.jpg", img)
		    queryimage=cv2.imread('/home/tanushri/Documents/SMAI/project/queryimage.jpg',4)
		    '''size = 1440, 1080
		    img = Image.open('queryimage.jpg')
		    img.thumbnail(size, Image.ANTIALIAS)
		    #img = img.resize((1440,1080), Image.ANTIALIAS)

		    img.save('q.jpg', "PNG")
		    #time.sleep(5)
		    #image = Image.open('q.jpg')'''
		    img = cv2.resize(queryimage, (1080, 1440)) 
		    #thumbnail = cv.CreateMat(1440, 1080, cv.CV_8UC3)
		    #cv.Resize(queryimage, thumbnail)
		    cv2.imwrite('q.jpg',img)
		    image=cv2.imread('/home/tanushri/Documents/SMAI/project/q.jpg',4)
		    index.write(ID)
		    image = np.mean(np.double(image),axis=2)
		    for i in range(0,4):
			for j in range(0,3):
		    		feaArrSingle = extractor.process_image(image[i*360:i*360+360,j*360:j*360+360])
		
		
				min=np.amin(feaArrSingle)
				max=np.amax(feaArrSingle)
		
		
				for k in range(0,128):
			
					feaArrSingle[0][k]=float(feaArrSingle[0][k]-min)/(max-min)
				for k in range(0,128):
					index.write(","+str(feaArrSingle[0][k]))
				
				#index.write(feaArrSingle[0])
		    index.write('\n')
		except:
			print 'yes'
			continue
	
    index.close()
		

   # '''pyplot.figure()
   # pyplot.plot(feaArr[0],'r')
    #pyplot.plot(feaArrSingle,'b')
   # pyplot.show()'''
#myntra_2977.jpg
#myntra_2999.jpg
#myntra_1541.jpg
#myntra_1991.jpg


