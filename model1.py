#!/usr/bin/python

import os
import numpy as np
import dischargingRate as dr
import screenParse as sc
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
import statistics as stats
from mpl_toolkits.mplot3d import Axes3D


class Modelling:
	def __init__(self,num=1):
		if num > 1:
			self.one = False
		else:
			self.one = True
		self.all_dump = dr.doInPool(num)
		#self.all_scr = sc.doInPool(num)
		self.X = []
		self.Y = []

	def computeFeatures(self):
		dict_ = {}
		sc_ = {}
		for i in range(len(self.all_dump)):
			dev = next(iter(self.all_dump[i]))
			dict_ = self.all_dump[i][dev]
			#sc_ = self.all_scr[i][dev]
			sc_ = sc.overlapDischarge(dev, dict_)
			break
			#func(dev, dict_, sc_)
		print(len(sc_.keys()), len(dict_.keys()))
		#list of features: 	time_in_mins_left_to_reach_low  [low <= 15]
		#			fg_usage_till_tNow
		#			fg_frac_till_tNow
		#			battery_level_now
		#			level_drop_till_tNow
		#			avg_discharge_rate_last_hour
		#			fg_frac_last_hour
		#look only into sessions from when fg logging started
		all_ = []
		start_t = sorted(sc_.keys())[0]
		sortedD = sorted(dict_.keys())
		for i in range(len(sortedD)):
			if sortedD[i] < start_t:
				continue
			#calculate total time_in_mins_left_to_reach_low [low <= 15]
			bL = dict_[sortedD[i]][-1][1]
			if bL > 15:
				last_event = self.extrapolate(dict_[sortedD[i]])
			else:
				last_event = dict_[sortedD[i]][-1]
			session = dict_[sortedD[i]]
			s = 0
			fg_now = 0
			hr_start = [session[0][0],session[0][1]]
			for j in range(len(session)):
				level_now = session[j][1]
				level_drop_now = session[0][1] - session[j][1]
				t_left_mins = round((last_event[0]-session[j][0]).total_seconds()/60,2)
				#fg_usage_till_tNow
				if sortedD[i] in sorted(sc_.keys()):
					#print(len(sc_[sortedD[i]]))
					for k in range(s, len(sc_[sortedD[i]])):
						if session[j][0] < sc_[sortedD[i]][k][0]:
							print('*',s,k, session[j][0], sc_[sortedD[i]][k][0],'---',fg_now)
							s = k
							break
						if session[j][0] > sc_[sortedD[i]][k][1]:
							fg_now += (sc_[sortedD[i]][k][1] - sc_[sortedD[i]][k][0]).total_seconds()/60
						elif sc_[sortedD[i]][k][0] < session[j][0] < sc_[sortedD[i]][k][1]:
							fg_now += (sc_[sortedD[i]][k][1] - sessio
						else:
							fg_now += (session[j][0] - sc_[sortedD[i]][k][0]).total_seconds()/60
							s = k
							break
				#print(sortedD[i], s, j,  fg_now)
				if j == 0:
					fg_frac_now = 0
				else:
					fg_frac_now = round(fg_now/((session[j][0] - session[0][0]).total_seconds()/60), 4)
				if fg_frac_now > 1:
					print(fg_now, ((session[j][0] - session[0][0]).total_seconds()/60), session[j][0], sc_[sortedD[i]][s])
				time_passed_now = int((session[j][0] - session[0][0]).total_seconds()/60)/10 * 10
				all_.append([level_now, level_drop_now, time_passed_now, fg_frac_now, t_left_mins])
			if i > 60:
				break
		
		all_sorted = sorted(all_, key=lambda x:x[2])
		X = list(map(lambda z: z[:4], all_sorted))
		Y = list(map(lambda z: z[4],all_sorted))
		outputs = []
		done = []
		for i in range(len(X)):
			#print(X[i],'------------->',Y[i])
			e = X[i]
			if e in done or e[3] == 0:
				continue
			outputs.append([])
			for j in range(i,len(X)):
				if e[2] > X[j][2]:
					break
				if e[0] == X[j][0] :
					outputs[-1].append(Y[j])
			done.append(e)
			#if i > 400:
			#	break
		#for i in range(len(done)):
		#	if len(outputs[i]) > 2:
		#		print(done[i], '---->', max(outputs[i]),min(outputs[i]),np.mean(outputs[i]))
		#	if done[i][2] > 100:
		#		break
		print(len(X), len(done))
		return True

	def extrapolate(self,list_):
		avg_rate = (list_[0][1] - list_[-1][1])/((list_[-1][0] - list_[0][0]).total_seconds()/60)
		new_t = list_[-1][0] + timedelta(minutes=int((list_[-1][1] - 15)/avg_rate))
		event = [new_t, 15, avg_rate]
		return event

	def connectedSteps(self, t_now, output):
		graph = {}

	def clustering(self,):
		features = []
		for i in range(len(self.X)):
			features.append([X[0],X[1],X[2],Y[0]])
		data = scale(features)
		reduced_data = PCA(n_components=2).fit_transform(data)
		kmeans.fit(reduced_data)
		centroids = kmeans.cluster_centers_
		labels_ = kmeans.labels_
		label_color = [color_map[l] for l in labels_]
		fig, ax = plt.subplots()
		ax.scatter(reduced_data[:,0], reduced_data[:,1], reduced_data[:,2], c=label_color, alpha=0.5)
		ax.plot(centroids[:,0], centroids[:,1], centroids[:,2],'k*')
		fig.show()
