#!/usr/bin/env python2
import os
import sys
import numpy as np
from numpy import vstack,array
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from pylab import plot, show
from datetime import *
from collections import defaultdict
import pickle
from scipy.cluster.vq import kmeans,vq

devices = []

def make_data():
    try:
        dict_ = pickle.load(open('/home/anudipa/Documents/Jouler_Extra/All_Data.p', 'rb'))
    except:
        print path
        return None 
    temp = []
    for dev in dict_:
	entries = 0
	s = sum(len(dict_[dev][days]) for days in dict_[dev])
	#print dev, s
#checking if it shd be filtered 
	if len(dict_[dev].keys()) < 7 or s < 100:
	    continue
	for day in dict_[dev]:
	    if day > 5:
		continue
	    for val in dict_[dev][day]:
		if val[0] < 0 or val[3] < 0:
		    print dev, val
		if val[0] < 60*20 or val[0]> 54000:
		    continue
	    	temp.append([val[0]/60,val[3]])
	    	entries += 1
	if entries > 0:
	    devices.append([dev, entries])
    return temp

def groupsForDevice(idx, data_):
    i = 0
    groups = {} 
    groups[0] = []
    groups[1] = []
    groups[2] = []
    total = 0
    for x in devices:
	total += x[1]
	count = 0
	while i < (total - 1):
	    if idx[i] == 0:
		count += 1
	    i += 1
	print x[0], count, x[1] 
	perc = (count/float(x[1]))*100.0
	if perc > 75:
	    groups[0].append(x[0])
	elif perc < 25:
	    groups[1].append(x[0])
	else:
	    groups[2].append(x[0])
	count = 0

    return groups

if __name__ == '__main__':
    data_ = array(make_data())
    if data_ is None:
	exit(0)
    #print devices
    centroids, _ = kmeans(data_, 2)
    idx, _ = vq(data_, centroids)
    groups = groupsForDevice(idx, data_)
    plot(data_[idx==0,0], data_[idx==0,1],'ob',
	 data_[idx==1,0], data_[idx==1,1],'or')
    plot(centroids[:,0], centroids[:,1], 'sg', markersize=10)
    print '0', len(groups[0])
    print '1', len(groups[1])
    print '2', len(groups[2])
    show()
