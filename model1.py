#!/usr/bin/python

import os
import numpy as np
import pandas as pd
import dischargingRate as dr
import screenParse as sc
from collections import defaultdict, Counter, OrderedDict
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
		print('sortedD',len(sortedD))
		self.test = defaultdict(list)
		count = 0
		fg_c =[0,0]
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
			last_ =  session[0][0]
			for j in range(len(session)):
				level_now = session[j][1]
				level_drop_now = session[0][1] - session[j][1]
				t_left_mins = round((last_event[0]-session[j][0]).total_seconds()/60,2)
				if j > 0:
					if session[j][0] == session[j-1][0]:
						#print('Duplicate', j, session[j], session[j-1])
						continue
				#fg_usage_till_tNow
				if sortedD[i] in sorted(sc_.keys()):
					#print(len(sc_[sortedD[i]]))
					for k in range(s, len(sc_[sortedD[i]])):
						#if datetime.datetime(2015, 4,11,6,0,0)<session[j][0]<datetime.datetime(2015, 4,11,20,0,0):
						#	print('**',fg_now, last_, session[j][0])
						if session[j][0] < sc_[sortedD[i]][k][0]:
							#s = k
							break
						if s == k and j >0 and sc_[sortedD[i]][k][0] < last_ < sc_[sortedD[i]][k][1]:
							if last_< session[j][0] < sc_[sortedD[i]][k][1]:
								fg_now += (session[j][0] - last_).total_seconds()/60
								#print(sortedD[i], sc_[sortedD[i]][k], last_, session[j][0])
								#last_ = session[j][0]
								#s = k
								break
							else:
								fg_now += (sc_[sortedD[i]][k][1] - last_).total_seconds()/60
						elif sc_[sortedD[i]][k][0]< session[j][0] < sc_[sortedD[i]][k][1]:
							fg_now += (session[j][0] - sc_[sortedD[i]][k][0]).total_seconds()/60
						elif session[j][0] > sc_[sortedD[i]][k][1]:
							fg_now += (sc_[sortedD[i]][k][1] - sc_[sortedD[i]][k][0]).total_seconds()/60
#						if datetime.datetime(2015, 5, 30,6,0,0)<session[j][0]<datetime.datetime(2015,5,30,20,0,0):
#							print('**',fg_now, '*',j,s,k,len(sc_[sortedD[i]]),last_, session[j][0], '||', sc_[sortedD[i]][k])
					if session[j][0] > sc_[sortedD[i]][k][1]:
						s = k+1
					else:
						s = k
					last_ = session[j][0]
						
				#print(sortedD[i], s, j,  fg_now)
				if j == 0:
					fg_frac_now = 0
				else:
					fg_frac_now = round(fg_now/((session[j][0] - session[0][0]).total_seconds()/60), 4)
				if fg_frac_now > 1:
					print('!!!', fg_now, ((session[j][0] - session[0][0]).total_seconds()/60), session[j][0])
					continue
				if fg_frac_now > 0.95:
					fg_c[0] += 1
				else:
					fg_c[1] += 1
				time_passed_now = int((session[j][0] - session[0][0]).total_seconds()/60)/10 * 10
				if i <= 300:
					all_.append([level_now, level_drop_now, time_passed_now, fg_frac_now, t_left_mins])
				else:
					self.test[sortedD[i]].append([level_now, level_drop_now, time_passed_now, fg_frac_now, t_left_mins])
					count += 1
				
			#if i > 300:
			#	break
		print('%%%%%', count, len(all_))
		print('!!!', fg_c)
		all_sorted = sorted(all_, key=lambda x:x[2])
		
		self.buildSteps(all_sorted)
		
		return True

	def extrapolate(self,list_):
		avg_rate = (list_[0][1] - list_[-1][1])/((list_[-1][0] - list_[0][0]).total_seconds()/60)
		new_t = list_[-1][0] + timedelta(minutes=int((list_[-1][1] - 15)/avg_rate))
		event = [new_t, 15, avg_rate]
		return event


	def clustering(self):
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

	def buildSteps(self,data_):
	#T, L, FG, dL : T (round by 5 mins) L (bin by 5) FG( .1), dL( by 5 level drop)
	#all_.append([level_now, level_drop_now, time_passed_now, fg_frac_now, t_left_mins])
		print('Building steps', len(data_))
		row_list = []
		for i in range(len(data_)):
			rem = data_[i][2]%60
			if rem == 0:
				T = data_[i][2]
			if rem < 30:
				T = int(data_[i][2]/60)*60 + 30
			else:
				T = int(data_[i][2]/60)*60 + 60
			if data_[i][0]%5 == 0:
				L = data_[i][0]
			else:
				L = int(data_[i][0]/5)*5 + 5
			if data_[i][1]%5 == 0:
				dL = data_[i][1]
			else:
				dL = int(data_[i][1]/5)*5 + 5
			if data_[i][3] <= 0.01:
				fg = 0.01
			elif data_[i][3] < 1.0:
				fg = (int((data_[i][3]+0.1)*10)/10)
			
			else:
				fg = 1.0
			target = data_[i][4]
			d = OrderedDict()
			d = {'A_time_from_start_mins':T, 'B_battery_level': L, 'C_drop_now': dL, 'D_foreground_frac': fg, 'E_discharge_time_left_mins': target}
			row_list.append(d)
		df = pd.DataFrame(row_list)
		df.sort_values(['A_time_from_start_mins','B_battery_level','C_drop_now','D_foreground_frac'], inplace=True, ascending=True)
		self.mothership = df
		#print(df.head(100))
		print('DataFrame created')
		return True

	def learning(self):
		main = self.mothership
		#reindex the dataframe, for same column A to D, get range for F
		#in new dataframe add A to D, for F add mean, lower and upper limits
		aggregations = {
			'E_discharge_time_left_mins': {
				'lower_quantile': lambda x: np.percentile(x, q=25),
				'mean_value': 'mean',
				'upper_quantile': lambda x: np.percentile(x, q=75),
				'standard_dev': 'std'
			}
		}
		#trial df with aggregated values for A <= 60
		new_df = main.groupby(['A_time_from_start_mins','B_battery_level','C_drop_now','D_foreground_frac']).agg(aggregations)
		print(new_df.columns)
		#print(new_df.index)
		return new_df
		#return main.groupby(['A_time_from_start_mins','B_battery_level','C_drop_now','D_foreground_frac'])['E_discharge_time_left_mins'].count()
		

	def predicting(self, session):
		#this will take a discharging session, and compare prediction and real values for 1 hour interval
		#([level_now, level_drop_now, time_passed_now, fg_frac_now, t_left_mins])
		grouped_ = self.learning()
		steps = []
		
		if len(session) > 5:
			for i in range(len(session)):
				if session[i][2]%60 < 30:
					T = int(session[i][2]/60)*60 + 30
				else:
					T = int(session[i][2]/60)*60 + 60
				if session[i][0]%5 == 0:
					L = session[i][0]
				else:
					L = int(session[i][0]/5)*5 + 5
				if session[i][1]%5 == 0:
					dL = session[i][1]
				else:
					dL = int(session[i][1]/5)*5 + 5
				if session[i][3] <= 0.01:
					fg = 0.01
				elif session[i][3] < 1.0:
					fg = (int((session[i][3]+0.1)*10)/10)
				else:
					fg = 1.0
				target = session[i][4]
				steps.append([T,L,dL,fg,target])
			last_ = steps[0]
			last_pred = 0
			val = []
			obv = []
			err = []
			t = []
			#algo: for same T, get output for all input parameters, take mean and compare with last prediction, if it increases too much then discard the prediction and update it by decreasing it .
			for i in range(len(steps)):
				if last_[0] == steps[i][0]:
					level = steps[i][1]
					#row = grouped_[(grouped_.index.get_level_values('A_time_from_start_mins')==30)]
					row = grouped_[(grouped_.index.get_level_values('A_time_from_start_mins')==steps[i][0]) & (grouped_.index.get_level_values('B_battery_level')==steps[i][1]) & (grouped_.index.get_level_values('C_drop_now')==steps[i][2]) & (grouped_.index.get_level_values('D_foreground_frac')==steps[i][3])]
					if len(row['E_discharge_time_left_mins','mean_value'].values) == 0:
						row =  grouped_[(grouped_.index.get_level_values('A_time_from_start_mins')==steps[i-1][0]) & (grouped_.index.get_level_values('B_battery_level')==steps[i-1][1])]
					#print(row['E_discharge_time_left_mins','mean_value'].values, '-----', val)
					val += list(row['E_discharge_time_left_mins','mean_value'].values)
					obv.append(steps[i][4])
					#print(row['E_discharge_time_left_mins','mean_value'].values, '-----', val)
				if i < len(steps)-1 and last_[0] != steps[i+1][0]:
					mean_p = np.mean(val)
					mean_o = np.mean(obv)
					if last_pred > 0 and (mean_p - last_pred) > 60:
						mean_p = last_pred + 30
					print('@T = ', last_[0],'predicted --->',mean_p, 'obeserved: ',mean_o)
					err.append(abs(mean_p - mean_o))
					t.append(last_[0])
					val = []
					obv = []
					last_ = steps[i+1]
					last_pred = mean_p

		return [err,t]

	def helpMe(self):
		#help me choose a session to predict or any other thing
		c = [0,0]
		dict_ = self.test
		#print(len(dict_.keys()))
		sortedD = sorted(dict_.keys())
		for d in range(len(self.all_dump)):
			dev = next(iter(self.all_dump[d]))
			dump = self.all_dump[d][dev]
			break
		print(len(dump.keys()))
		fig, ax = plt.subplots()
		for i in range(len(sortedD)):
			session = dump[sortedD[i]]
			span = (session[-1][0] - session[0][0]).total_seconds()
			start_  = session[0][1]
			end_ = session[-1][1]
			if span >= 20*3600 and end_ <= 15:
				d_ = self.predicting(dict_[sortedD[i]])
				print('-------------------------', i)
				ax.plot(d_[1][2:], d_[0][2:],'k')
				ax.plot(d_[1][2:], d_[0][2:], 'ro')
				c[1] += 1
			else:
				c[0] += 1
#			elif span > 18*3600:
#				c[0] += 1
#			elif span < 5*3600:
#				c[0] += 1
		ax.set_xlabel('Time since start (mins)')
		ax.set_ylabel('Error in mins')
		ax.set_title('For sessions more than 20 hours')
		fig.show()
		print(i,c)
		return False


#pseudocode:
#For the present discharging session
# 1. for current time t, get <P1 = t_since_start, P2= level_now, P3=total_drop_since_start, P4=frac_of_fg_duration>
# 2. lookup main dataframe for similar entry with <P1,P2,P3,P4>
# 	2a. if present bump w1 [weight associated with the observation]
#	2b. if present add observation to projected_time_left[]
#	2c. if not present add -1 to projected_time_left[]
# 3. continue 1 & 2 till end of discharge session
# 4. compute backwards observed_time_left
# 5. compare each entry in observed_time_left and projected_time_left
#	5a. for error less than 30 mins increase w2 [weight associated with <P1,P2,P3,P4>--<Target>]
#	5b. for error more than 60 mins decrease w2
# 6. if <P1,P2,P3,P4>--<Target> present in table, update new weights
# 7. after every 7 days update the most_confident_paths dataframe  [decide threshold for w1,w2 to be included]
	def feedBack(self):
	
		return False		
