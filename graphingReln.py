#!/usr/bin/env python

import os
import numpy as np
from multiprocessing import Process, current_process, Pool
from collections import defaultdict, Counter
import pickle
from datetime import *
from pylab import *
import matplotlib.pyplot as plt
from scipy.stats import itemfreq
from scipy.cluster.vq import *
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn import utils, preprocessing
import testCases
import dischargingRate as dR
import statistics as stats
from mpl_toolkits.mplot3d import Axes3D


#the generic pool function
def doInPool():
	all_dump = dR.doInPool()
	for d in range(len(all_dump)):
		dev = next(iter(all_dump[d]))
		dict_ = all_dump[d][dev]
		print(dev, 'no of sessions: ', len(dict_.keys()))
		startbatteryData(dict_, dev)
def temp(all_dump):
	for d in range(len(all_dump)):
		dev = next(iter(all_dump[d]))
		dict_ = all_dump[d][dev]
		print(dev, 'no of sessions: ', len(dict_.keys()))
		#startbatteryData(dict_, dev)
		clusterFeatures(dict_,dev)
		break

#for different battery level start, how does other features differ
def startbatteryData(dict_, dev):
	#get start levels for each session, span, total drop in charge, variance in discharging rate
	data1 = defaultdict(list)
	data2 = defaultdict(list)
	sortedK = sorted(dict_.keys())
	if len(sortedK) < 2:
		print('Too small, data missing!!', dev)
		return
	for i in range(len(sortedK)):
		if not (0 < sortedK[i].weekday() < 6):
			continue
		if len(dict_[sortedK[i]]) < 2:
			print(dev, sortedK[i], 'Missing Data', dict_[sortedK[i]])
			continue
		start_level 	= dict_[sortedK[i]][0][1]
#		total_drop	= (dict_[sortedK[i]][0][1] - dict_[sortedK[i]][-1][1])
		drop		= dict_[sortedK[i]][-1][1]
		span_in_min	= (dict_[sortedK[i]][-1][0] - dict_[sortedK[i]][0][0]).total_seconds()/60.0
		rates_		= [dict_[sortedK[i]][j][2] for j in range(1, len(dict_[sortedK[i]]))]
		var_rate	= stats.pvariance(rates_)
		data1[start_level].append([drop, span_in_min, var_rate])
		data2[int(start_level/10)].append([drop/10, span_in_min, var_rate])
	#check data
	#bin 10
	for bins in sorted(data2.keys()):
		if len(data2[bins]) < 2:
			continue
		drop = [data2[bins][i][0] for i in range(len(data2[bins]))]
		span = [data2[bins][i][1] for i in range(len(data2[bins]))]
		var_rate = [data2[bins][i][2] for i in range(len(data2[bins]))]
#		print(len(drop),drop)
		print('Bin:', bins, '-----------> std dev drop:', stats.stdev(drop), '; std dev span:', stats.stdev(span), '; mean variance:', stats.mean(var_rate))


def clusterFeatures(dict_, dev):
#get all features: start_level, end_level, level_1st_hr, span, high usage in which quarter
	features = []
	sortedK = sorted(dict_.keys())
	for i in range(len(sortedK)):
		if not (0 < sortedK[i].weekday() < 6):
			continue
		if len(dict_[sortedK[i]]) < 2:
			print(dev, sortedK[i], 'Missing Data', dict_[sortedK[i]])
			continue
		start_level = dict_[sortedK[i]][0][1]
		end_level = dict_[sortedK[i]][-1][1]
		hr_day = sortedK[i].hour
		lvl_first_hr = -1
		t1 = dict_[sortedK[i]][0][0]
		for j in range(1,len(dict_[sortedK[i]])):
			t2 = dict_[sortedK[i]][j][0]
			if 3600 <= (t2 - t1).total_seconds() < 2*3600:
				lvl_first_hr = dict_[sortedK[i]][j][1]
				break
			elif (t2 - t1).total_seconds() > 2*3600:
				lvl_first_hr = dict_[sortedK[i]][0][1]
				break
		span = (dict_[sortedK[i]][-1][0] - dict_[sortedK[i]][0][0]).total_seconds()/60.0
		if span > 36 *60:
			mid = int(len(dict_[sortedK[i]])/2)
			print(dict_[sortedK[i]][0], dict_[sortedK[i]][-1], len(dict_[sortedK[i]]), dict_[sortedK[i]][mid])
			continue
		sess_high = getHighUsage(dict_[sortedK[i]])
		features.append([start_level, end_level, hr_day, lvl_first_hr, span, sess_high])
	
	for i in range(len(features)):
		if features[i][5] == -1:
			print(features[i])		
		
	#PCA & KMeans
	data = scale(features)
#	pca_reduced = PCA(n_components=3).fit(data)
	reduced_data = PCA(n_components=2).fit_transform(data)
#	kmeans = KMeans(init=pca_reduced.components_, n_clusters=3, n_init=1)
	kmeans = KMeans(init='k-means++', n_clusters=3, n_init=10)
#	kmeans.fit(data)
	kmeans.fit(reduced_data)
	print(kmeans.labels_)
	print('---------------------------------------')
	print(kmeans)
	centroids = kmeans.cluster_centers_
	labels_ = kmeans.labels_
	color_map = { 0 : 'r', 1 : 'b', 2: 'g'}
	label_color = [color_map[l] for l in labels_]
#	fig, ax = plt.subplots()
#	ax.scatter(reduced_data[:,0], reduced_data[:,1], c=label_color, alpha=0.5)
#	ax.plot(centroids[:,0], centroids[:,1], 'k*')
#	ax = Axes3D(fig, elev=48, azim=134)
#	ax.scatter(reduced_data[:,0], reduced_data[:,1], reduced_data[:,2], alpha = 0.5)
#	ax.plot(centroids[:,0], centroids[:,1], centroids[:,2], 'k*')
#	fig.show()
	lookIntoClusters(features, labels_)

def lookIntoClusters(data, labels_):
	no_of_clusters = 3
	cluster = [[],[],[]]
	for i in range(len(labels_)):
		cluster[labels_[i]].append(data[i])
	#fig, (ax1, ax2, ax3) = plt.subplots(1,3, sharex=True)
	#6 columns = start, end, hr, lvl_1st_hr, span, high
	for i in range(3):
		temp = [[cluster[i][j][0], cluster[i][j][2], cluster[i][j][3]] for j in range(len(cluster[i]))]
		data = []
		target = []
		for j in range(len(cluster[i])):
			data.append([cluster[i][j][0], cluster[i][j][2], cluster[i][j][3]])
			lvl = cluster[i][j][1]
			span = cluster[i][j][4]
			new_lvl = int(lvl/10)*10
			if span < 100:
				new_span = int(span/100)*10
			target.append([int(cluster[i][j][1]),int(cluster[i][j][4])])
		print(utils.multiclass.type_of_target(target))
		mid = int(len(data)/2)
		knn = KNeighborsClassifier()
		multi_target_ = MultiOutputClassifier(knn, n_jobs=1)
		multi_target_.fit(data[:mid], np.array(target[:mid]))
#		print('***',multi_target_.predict(data[mid+1:]))
#		print('!!!',target[mid+1:])
		print(multi_target_.score(data[mid+1:], np.array(target[mid+1:])))			 
		
#		fig, (ax1, ax2, ax3) = plt.subplots(1,3, sharex=True)
#		ax1.boxplot(data1,sym='',whis=[25,75],patch_artist = True)
#		ax1.set_ylabel('Span in minutes')
#		ax2.boxplot(data2, sym='',whis=[25,75],patch_artist = True)
#		ax2.set_ylabel('Battery Level At End')
#		ax3.boxplot(data3, sym='',whis=[25,75],patch_artist = True)
#		ax3.set_ylabel('Span in minutes')
#		fig.show()




def getHighUsage(session):
	sess_len = (session[-1][0] - session[0][0]).total_seconds()
#	div = int(sess_len/4)
#	first_quarter_start = session[0][0]
#	second_quarter_start = session[0][0]+ timedelta(seconds=div)
#	third_quarter_start = session[0][0]+timedelta(seconds=div*2)
#	fourth_quarter_start = session[0][0]+timedelta(seconds=div*3)
#	l = [[],[],[],[]]
	l = [[],[]]
	first_start = session[0][0]
	second_start = session[0][0] + timedelta(seconds = int(sess_len/2))
	for i in range(len(session)):
		if first_start <= session[i][0] < second_start:
			l[0].append(session[i][2])
		else:
			l[1].append(session[i][2])
#		elif second_quarter_start <= session[i][0] < third_quarter_start:
#			l[1].append(session[i][2])
#		elif third_quarter_start <= session[i][0] < fourth_quarter_start:
#			l[2].append(session[i][2])
#		elif fourth_quarter_start <= session[i][0]:
#			l[3].append(session[i][2])
#	if len(l[0]) == 0 or len(l[1]) == 0 or len(l[2]) == 0 or len(l[3])==0:
#		print('Hmmmmm, wtf', sess_len, len(session), div, session[0][0], second_quarter_start, third_quarter_start, fourth_quarter_start, session[-1][0])
#		print('****', len(l[0]), len(l[1]), len(l[2]), len(l[3]))
#	high =[np.mean(l[0]), np.mean(l[1]), np.mean(l[2]), np.mean(l[3])]
	if len(session) == 2:
		rate = (session[0][1] - session[1][1])/((session[1][0] - session[0][0]).total_seconds()/60)
		l[0] = l[1] = rate

	high = [np.mean(l[0]), np.mean(l[1])]
	
	if max(high) == 0.0:
		print(l)
		print(session)
		return -1
	elif high.count(max(high)) == 2:
		return 2
	else:
		return high.index(max(high))


def chargingPredict(dict_, dev):
	chargeDt = list()
	x = []
	y = []
	z = []
	color_map = { 0 : 'r', 1 : 'b', 2: 'g'}
	sortedK = sorted(dict_.keys())
	count = 0
	for i in range(len(sortedK)-1):
		start_lvl = dict_[sortedK[i]][-1][1]
		start_hr  = dict_[sortedK[i]][-1][0].hour
		span 	  = (dict_[sortedK[i+1]][0][0] - dict_[sortedK[i]][-1][0]).total_seconds()/60
		if span < 10:
			#print('Hmmmmm, short session')
			count += 1
			continue
		if span > 15*60:
			#print('Hmmm, too long, overlooking')
			count += 1
			continue
		chargeDt.append([start_lvl, start_hr, span])
		x.append(start_lvl)
		y.append(start_hr)
		z.append(span)
	print( 'Overlooked: ', count)
	centroids, res = kmeans2(chargeDt, 3)
	#idx, res = vq(chargeDt, centroids)
	l = len(chargeDt)
	X = np.reshape(chargeDt, (l, 3))
	#f, (ax1, ax2) = plt.subplots(1,2)
	print(centroids)
	label_color = [color_map[l] for l in res]
	fig = figure(dpi = 300)
	ax = Axes3D(fig, elev=48, azim=134)
	ax.scatter(X[:,0], X[:,1], X[:,2], c=label_color, alpha = 0.5)
	ax.plot(centroids[:,0], centroids[:,1], centroids[:,2], 'k*')
	ax.set_xlabel('Level at Start', fontsize=4)
	ax.set_ylabel('Hour of Date', fontsize=4)
	ax.set_zlabel('Span in Mins', fontsize=4)
	#fig.set_size_inches(4.0, 3.5)
	fig.show()
