#!/usr/bin/env python

import os
from pandas import *
import numpy as np
from multiprocessing import Process, current_process, Pool
from collections import defaultdict
import pickle
import json
from datetime import *
from pylab import *


def convert(data):
	data_type = type(data)
	if data_type == bytes : return data.decode()
	if data_type in (str,int,float): return data
	if data_type in (datetime.datetime,datetime.date): return data
	if data_type == dict: data = data.items()
	return data_type(map(convert, data))


def computeTime(dev):
	file1 = '/home/anudipa/Documents/Jouler_Extra/discharge2/'+dev+'.p'
	file2 = '/home/anudipa/Documents/Jouler_Extra/charge2/'+dev+'.p'
	try:
		print('Starting')
		tmp1 = pickle.load(open(file1,'rb'), encoding='bytes')
		tmp2 = pickle.load(open(file2,'rb'), encoding='bytes')
		dDataset = convert(tmp1)
		cDataset = convert(tmp2)
		ddata = dDataset[dev]
		cdata = cDataset[dev]
	except Exception as e:
		print(e)
		return

	sortedD = sorted(ddata.keys())
	#final dictionaries
	w1 = defaultdict(list)				#key:batterylevel bins of 10
	w2 = defaultdict(list)
	#working dictionaries
	ddict_ = {}
	chargeAll = []
	last_ = ddata[sortedD[0]][0][0][0]
	for t in range(len(sortedD)):
		d = sortedD[t]
		day = d.weekday()
		listOfSessions = ddata[d]
		for i in range(len(listOfSessions)):
			eachSession = listOfSessions[i]
			eachCycle = []
			for k in range(len(eachSession)):
				levels = int(eachSession[k][0]/10)
				ttime = eachSession[k][1]
				ddict_[ttime] = levels
		if d not in cdata.keys():
			continue

		listOfSessions = cdata[d]
		if len(listOfSessions) == 0:
			continue
		for i in range(len(listOfSessions)):
			start_ =  listOfSessions[i][0]
			end_   = listOfSessions[i][2]
			if i > 0 and (start_ - endLast).total_seconds() < 300:
				chargeAll[-1][1] = end_
			else:
				chargeAll.append([start_, end_])
			endLast = end_
	sortedK = sorted(ddict_.keys())
	last = 0
	for i in range(len(chargeAll)):
		start_ = chargeAll[i][0]
		#print('charge', start_)
		for j in range(last,len(sortedK)):
			if sortedK[j] <= start_:
				level = ddict_[sortedK[j]]
				diff = (start_ - sortedK[j]).total_seconds()/60
				if level == 10 and diff < 10:
					#print(sortedK[j])
					continue
				if sortedK[i].weekday() in [1,2,3,4,5]:
					w1[level].append(diff)
				else:
					w2[level].append(diff)
			else:
				last = j
				break
	#print(chargeAll)
	print(w1.keys())
	return {'weekdays':w1, 'weekends':w2}

def doInPool():
	pFile = pickle.load(open('/home/anudipa/Documents/Jouler_Extra/master_list_100.p','rb'), encoding='bytes')
	filtered = convert(pFile)
	#filtered = filtered[:10]
	print(filtered)
	pool = Pool(processes=8)
	res = pool.map(computeTime, filtered)
	pool.close()
	pool.join()
	return res




print('*********')
allDays = computeTime('cdfce167c0f7fbcdc14b841f0cf4cd6d7fe6d470')
#all_dump = doInPool()
#print(len(all_dump))
#weekdays = defaultdict(list)
#weekends = defaultdict(list)
#for i in range(len(all_dump)):
#	each_w1 = all_dump[i]['weekdays']
	#print(len(each_w1[0]))
#	each_w2 = all_dump[i]['weekends']
#	for batterylevel in each_w1.keys():
#		weekdays[batterylevel] += (each_w1[batterylevel])
#	for batterylevel in each_w2.keys():
#		weekends[batterylevel] += (each_w2[batterylevel])

weekdays = allDays['weekdays']
fig, ax = plt.subplots()
data = []
N = len(weekdays.keys())
for batterylevel in weekdays.keys():
	data.append(weekdays[batterylevel])
bp = ax.boxplot(data, sym='', vert=0, whis=[25,75])
#ax.set_yticks(np.arange(0, 11, 1))
ax.set_yticklabels(np.arange(0, 110, 10))
ax.set_xlabel('Time to charge (mins)')
ax.set_ylabel('Batterylevel')
ax.set_title('Time left to charge on weekdays')
fig.show()
