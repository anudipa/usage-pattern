#!/usr/bin/python

import os
import numpy as np
import pandas as pd
import screenParse as sc
from collections import defaultdict, Counter, OrderedDict
import pickle
from datetime import *
from pylab import *
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM


def computeStates(path):
	#load the pickle
	try:
		pp = pickle.load(open(path, 'rb'), encoding='bytes')
		#print(len(pp['0c037a6e55da4e024d9e64d97114c642695c5434']['sessions']))
		dev = next(iter(pp))
		#print(dev, type(dev))
		days = pp[dev]['days']
		dict_ = pp[dev]['sessions']
	except Exception as e:
		print('Error in loading/accessing pickle!', e)
		return
	#load screen on/off data
	try:
		sc_ = sc.overlapDischarge(dev, dict_)
	except Exception as e:
		print('Error in loading/accessing screen data!', e)
		return
	debug1(dict_, sc_)
	#State = {T, FG, L, D}	Output = {T_Left}
	#	T_Left:	time in mins left to reach level <= low [ low = 15 ]
	#	T:	time in mins passed since start of the discharging sesion
	#	FG:	fraction of time for which screen was on since start of discharging session
	#	L:	battery level at current time
	#	D:	total battery level drop since start of discharging session

	print(len(sc_.keys()), len(dict_.keys()))			#debug

	all_states = []
	all_output = []
	sortedS = sorted(sc_.keys())
	sortedD = sorted(dict_.keys())
	start_t = sortedS[0]
	count = 0
	fg_c = [0,0]
	for i in range(len(sortedD)):
		if sortedD[i] < start_t:
			continue
		#calculate total time_in_mins_left_to_reach_low [low <= 15]
		lastL = dict_[sortedD[i]][-1][1]
		last_event = dict_[sortedD[i]][-1]
		if lastL > 15:
			last_event = extrapolate(dict_[sortedD[i]])
		if last_event is None:
			print(sortedD[i], i)
			continue
		session = dict_[sortedD[i]]		#al recorded events for this discharging session
		last_ = session[0][0]
		s = 0
		fg_now = 0
#		sequence = [[],[]]
		sequence = []
		for j in range(len(session)):
			level_now = session[j][1]
			level_drop_now = session[0][1] - session[j][1]
			t_left_mins = int((last_event[0]-session[j][0]).total_seconds()/60)
			if j > 0:
				if session[j][0] == session[j-1][0]:
					continue
			#fg_usage_till_tNow
			if sortedD[i] in sorted(sc_.keys()):
				for k in range(s, len(sc_[sortedD[i]])):
					if session[j][0] < sc_[sortedD[i]][k][0]:
						break
					if s == k and j >0 and sc_[sortedD[i]][k][0] < last_ < sc_[sortedD[i]][k][1]:
						if last_< session[j][0] < sc_[sortedD[i]][k][1]:
							fg_now += (session[j][0] - last_).total_seconds()/60
							break
						else:
							fg_now += (sc_[sortedD[i]][k][1] - last_).total_seconds()/60
					elif sc_[sortedD[i]][k][0]< session[j][0] < sc_[sortedD[i]][k][1]:
						fg_now += (session[j][0] - sc_[sortedD[i]][k][0]).total_seconds()/60
					elif session[j][0] > sc_[sortedD[i]][k][1]:
						fg_now += (sc_[sortedD[i]][k][1] - sc_[sortedD[i]][k][0]).total_seconds()/60
				if session[j][0] > sc_[sortedD[i]][k][1]:
					s = k+1
				else:
					s = k
				last_ = session[j][0]
			if j == 0:
				fg_frac_now = 0
			else:
				fg_frac_now = round(fg_now/((session[j][0] - session[0][0]).total_seconds()/60), 4)
			if fg_frac_now > 1:
				print('!!!', fg_now, ((session[j][0] - session[0][0]).total_seconds()/60), session[j][0])
				continue
			time_passed_now = int((session[j][0] - session[0][0]).total_seconds()/60)/10 * 10
		#####################################################################################
			L = level_now
			dL = level_drop_now
			T = time_passed_now
			FG = fg_frac_now
			T_Left = t_left_mins
			event = roundStates([T,L,dL,FG]) #+ [T_Left]
#			sequence[0].append(event)
#			sequence[1].append(T_Left)
			sequence.append(event)
			#all_states.append(roundStates([T,L,dL,FG]))
			all_output.append(T_Left)
			if L < 15:
				break
		all_states.append(np.array(sequence))
		######################################################################################
	all_ = {}
	all_['states'] = all_states
	all_['output'] = all_output
	return all_

def roundStates(data_):
	#data_ = [T,L,dL,FG]
	#T (round by 30 mins) L (bin by 5) FG( .1), dL( by 5 level drop)
	#1. round T
	rem = data_[0]%60
	if rem == 0:
		T = data_[0]
	elif rem  < 30:
		T = int(data_[0]/60)*60 + 30
	else:
		T = int(data_[0]/60)*60 + 60
	#2. round L
	if data_[1]%5 == 0:
		L = data_[1]
	else:
		L = int(data_[1]/5)*5 + 5
	#3. round dL or drop
	if data_[2]%5 == 0:
		dL = data_[2]
	else:
		dL = int(data_[2]/5)*5 + 5
	#4. round FG or foreground usage fraction
	if data_[3] <= 0.01:
		FG = 0.01
	elif data_[3] < 1.0:
		FG = (int((data_[3]+0.1)*10)/10)
	else:
		FG = 1.0
	return([T,L,dL,FG])



def extrapolate(list_):
	avg_rate = (list_[0][1] - list_[-1][1])/((list_[-1][0] - list_[0][0]).total_seconds()/60)
	if avg_rate == 0.0:
		print('Weird!',list_[0], list_[-1], len(list_))
		return None
	new_t = list_[-1][0] + timedelta(minutes=int((list_[-1][1] - 15)/avg_rate))
	event = [new_t, 15, avg_rate]
	return event



######################################################
############  HMM IMPLEMENTATION  ####################
######################################################
def hmmModel(states, output):
	data = states[:2]
	for
	print(np.array(data).shape)
	X = np.array(data).reshape(4,)
	lengths = [len(X[i]) for i in range(len(X))]
	print(X)
	model = GaussianHMM().fit(X,lengths)
	print(model.transmat_)
	return


def debug1(dict_, sc_):
	#debug:			matching
	sortedS = sorted(sc_.keys())
	sortedD = sorted(dict_.keys())
	k = 0 
	c = 0
	err = 0
	for i in range(len(sortedS)):
		for j in range(k,len(sortedD)):
			if sortedS[i] == sortedD[j]:
				c += 1
				k = j
				d1 = 0
				d2 = (dict_[sortedD[j]][-1][0] - dict_[sortedD[j]][0][0]).total_seconds()
				for x in range(len(sc_[sortedS[i]])):
					d1 += (sc_[sortedS[i]][x][1] - sc_[sortedS[i]][x][0]).total_seconds()
				if d1 > d2:
					err += 1
					print(sortedD[j], 'differ by', d1-d2, 'seconds')
				break
		if j > len(sortedD) - 2:
			print('reached end')
			break
	print('Total matches: ', c, 'error', err)

if __name__ == "__main__":
	path = '/home/anudipa/Documents/Jouler_Extra/scripts/data/shortlisted/0f73f649f1e0bb371f1fdcadf86f02567670315a.p'
	all_ = computeStates(path)
	if all_ is not None:
		print('Input states', len(all_['states']))
	hmmModel(all_['states'], all_['output'])
