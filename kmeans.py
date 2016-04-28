#!/usr/bin/python
import csv,math,random
from collections import defaultdict
import numpy as np
import json
import matplotlib.pyplot as plt
def calculateCentroid(pts):
		numPoints = len(pts)
		coords = [p for p in pts]
		unzipped = zip(*coords)
		centroid_coords = [math.fsum(dList)/numPoints for dList in unzipped]
		return centroid_coords

def getdist(a, b):
	    s=0.0
	    #print len(b),len(a)
	    
	    for i1 in range(0,len(b)):
		s=s+(a[i1]-b[i1])*(a[i1]-b[i1])
	
	 
	    return math.sqrt(s)

def update(i, pts):
		old_centroid = centroids[i]
		centroids[i] = calculateCentroid(pts)
		shift = getdist(old_centroid, centroids[i]) 
		return shift

def kmeans(filename,nc):


	global centroids,l	
	centroids= random.sample(points.values(), numofclusters)
	cutoff=0.5
	
	
	
	itr=0
	while 1:
		l = [ {} for c in centroids]
		
		itr+=1
		for c in points.keys():
			p=points[c]
			smallestdist=getdist(p, centroids[0])
			index=0
			for j in range(0,numofclusters):
				distance = getdist(p, centroids[j])
				if distance < smallestdist:
		            		smallestdist = distance
		            		index = j
		        l[index][c]=p
			
			
		bs=0.0
		for i in range(numofclusters):
			if len(l[i])>0:
		   	 	shift = update(i,l[i].values())
		    	 	bs = max(bs, shift)
		if bs < cutoff :
		    print "Converged after %s iterations" % itr
		    break
	j=json.dumps(centroids,separators=(',',':'))
	
	f.write(j+'\n')
	j=json.dumps(l,separators=(',',':'))
	f2.write(j+'\n')
	print centroids
	#print len(l[0]),len(l[1])

	#http://www.imm.dtu.dk/~perbb/MAS/ST116/module02/index.html
	#http://scikit-learn.org/stable/modules/clustering.html
	#http://datamining.rutgers.edu/publication/internalmeasures.pdf
	#https://en.wikipedia.org/wiki/Silhouette_%28clustering%29
columns = defaultdict(list) # each value in each column is appended to a list

with open('sift2.txt') as f:
    reader = csv.DictReader(f,delimiter=',') # read rows into a dictionary format
    for row in reader: # read a row as {column1: value1, column2: value2,...}
	for (k,v) in row.items(): # go over each column name and value 
	  columns[k].append(v)
numofclusters=100
x=len(columns.keys())-2
l=[]
labels=[]
labels2=[]
flag=0
key=columns.keys()
for i in range(0,numofclusters):
	l.append([])
ln=len(columns['1'])

points={}
tmp=[]
centroids=[]
f=open('centroids4.txt','a')
f2=open('results3.txt','a')
for i in range(1,1537,128):
	print i
	for k in range(0,ln):
		for j in range(i,i+128):
			try:
				tmp.append(float(columns[str(j)][k]))
			except:
				
				continue
		if len(tmp)==128:
			points[columns['0'][k]]=tmp
		else:
			print 'yes'
			
		tmp=[]
	kmeans('sift2.txt',3)
	tmp=[]
	points={}

	
	
f.close()
f2.close()
