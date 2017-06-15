#!/usr/bin/env python

import os
from pandas import *
import numpy as np
from multiprocessing import Process, current_process, Pool
from collections import defaultdict, Counter
import pickle
import json
from datetime import *
from pylab import *
import timeit
import dischargingRate as dr

path1 = '/home/anudipa/pattern/pickles/Screen/'
path_to_dev = '/home/anudipa/pattern/master_list_100.p'
path2 = '/home/anudipa/Documents/Jouler_Extra/pickles/Screen/'
path_to_dev2 = ''



def convert(data):
	data_type = type(data)
	if data_type == bytes : return data.decode()
	if data_type in (str,int,float): return data
	if data_type in (datetime.datetime,datetime.date): return data
	if data_type == dict: data = data.items()
	return data_type(map(convert, data))

def doInPool(x):
	pFile = pickle.load(open(path_to_dev,'rb'), encoding='bytes')
	filtered = convert(pFile)
	filtered = filtered[:x]
	print(filtered)
	pool = Pool(processes = 4)
	res = pool.map(parseScreen, filtered)
	pool.close()
	pool.join()
	return res


def parseScreen(dev):
	file1 = path2+dev+'.p'
#	file1= '/home/anudipa/pattern/pickles/Screen/0c037a6e55da4e024d9e64d97114c642695c5434.p'
	try:
		print('Starting', dev)
		tmp1 = pickle.load(open(file1, 'rb'), encoding='bytes')
		scdata = convert(tmp1)
		#scdata = tmp1[dev]
	except Exception as e:
		print('Error while processing file', e)
		return
	#dict_ = defaultdict(list)
	allSessions = {}
	print(len(scdata.keys()))
	sortedD = sorted(scdata.keys())
	flag = False
	start = sortedD[0]
	for i in range(len(sortedD)):
		dict_ = scdata[sortedD[i]]
	#get sessions for screen on and screen off : android.intent.action.SCREEN_ON, android.intent.action.SCREEN_OFF
#		if sortedD[i] > datetime.datetime(2015,4,5,0,0,0) and sortedD[i] < datetime.datetime(2015,4,6,23,0,0):
#			print(sortedD[i], dict_['Action'])
		if dict_['Action'] == 'android.intent.action.SCREEN_ON' and not flag:
			start = sortedD[i]
			flag = True
		elif flag and dict_['Action'] == 'android.intent.action.SCREEN_OFF':
			allSessions[start] = sortedD[i]
			flag = False
			#print(start,'--->', sortedD[i], dict_['Action'])
#			if (sortedD[i] - start).total_seconds() > 1*3600:
#				print(start,'--->', sortedD[i], (sortedD[i] - start).total_seconds()/3600)
		elif flag and dict_['Action'] == 'android.intent.action.SCREEN_ON':
			if (sortedD[i] - start).total_seconds() > 12*3600:
				start = sortedD[i]
	data = []
	c = 0
	for key in allSessions.keys():
		data.append((allSessions[key] - key).total_seconds())
		if (allSessions[key] - key).total_seconds() > 5*60*60:
			#print(key, '--->', allSessions[key])
			c += 1
	print(len(data), min(data), max(data), c)
	return allSessions

#get all foreground sessions for every discharge sessions: list
#get the battery levels for those fg sessions, and see when in fg session the battery level converges	
def overlapDischarge(dev, dict_):
	file1 = path2+dev+'.p'
	try:
		print('Starting now on', dev)
		tmp1 = pickle.load(open(file1, 'rb'), encoding='bytes')
		scdata = convert(tmp1)
	except Exception as e:
		print('Error while processing file', e)
#first get the discharge rates for this device
	screen_ = parseScreen(dev)
#	dict_ = dr.discharge(dev)
	sortedK = sorted(dict_.keys())
	k = 0
	sortedS = sorted(screen_.keys())
	print(len(sortedK), len(sortedS))
	overlap = defaultdict(list)
	for i in range(1,len(sortedK)):
		list_ = dict_[sortedK[i]]
#get the start and end of a discharge session, look for screen in sessions between those, keep track of the next screen on
#the datatype is dictionary containing list of [start,end] for each screen on session
		start_  = sortedK[i]
		end_ = dict_[sortedK[i]][-1][0]
		flag = False
		if k >= len(sortedS)-1:
			print(k, sortedS[-1], sortedK[i])
			break
		for j in range(k,len(sortedS)):
			#print(j, len(sortedS))
			#if j > 200:
			#	print(j,'Exit')
			#	return
			#print(j, flag, start_, sortedS[j], screen_[sortedS[j]], end_)
			if sortedS[j] < start_:
				#print(j,'****Continue')
				continue
			elif not flag and sortedS[j] >= start_ and sortedS[j] < end_:
				flag = True
			elif sortedS[j] > end_:
				#print('end of session', sortedS[j], end_)
				k = j
				break
			#print(start_, sortedS[j], screen_[sortedS[j]], end_)
			if flag:
				if screen_[sortedS[j]] < end_:
					overlap[start_].append([sortedS[j], screen_[sortedS[j]]])
				else:
					overlap[start_].append([sortedS[j],end_])
					#print('***Breaking')
					flag = False
					k = j
					break
		k = j
	count1 = 0
	count2 = 0
	d = []
	for key in sorted(overlap.keys()):
		num_sessions = len(overlap[key])
		fg_duration_sec = 0
		for i in range(len(overlap[key])):
			fg_duration_sec += (overlap[key][i][1] - overlap[key][i][0]).total_seconds()
		total_duration = (dict_[key][-1][0]-key).total_seconds()
		if fg_duration_sec > 4*total_duration/5:
			d.append(total_duration/60)
#			print('Start at ', key, ': ', num_sessions, total_duration/60, '--' , fg_duration_sec/60, ':',dict_[dev][key][-1][0])
#			for i in range(len(overlap[key])):
#				print(i, overlap[key][i][0], overlap[key][i][1])
			count2 += 1
		count1 += 1
	if len(d) > 0:
		print('****',count1, count2, np.mean(d))
	else:
		print('****',count1, count2)
#	sessions = {}
#	sessions[dev] = overlap
	return overlap
		

#print("Starting")
#overlapDischarge('0c037a6e55da4e024d9e64d97114c642695c5434')
