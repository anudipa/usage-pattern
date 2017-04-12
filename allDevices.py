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

#weekdays = defaultdict(list)
#weekends = defaultdict(list)

def convert(data):
        data_type = type(data)
        if data_type == bytes : return data.decode()
        if data_type in (str,int,float): return data
        if data_type in (datetime.datetime,datetime.date): return data
        if data_type == dict: data = data.items()
        return data_type(map(convert, data))


def discharge(dev):
	file1 = '/home/anudipa/Documents/Jouler_Extra/discharge2/'+dev+'.p'
	print(file1)
	try:
		print('Starting')
		tmp1 = pickle.load(open(file1,'rb'), encoding='bytes')
		dDataset = convert(tmp1)
		ddata = dDataset[dev]
	except Exception as e:
		print(e)
		return

	sortedD = sorted(ddata.keys())
	max_span = 0
	w1 = defaultdict(list)
	w2 = defaultdict(list)

	for d in sortedD:
		day = d.weekday()
		listOfSessions = ddata[d]
		for i in range(len(listOfSessions)):
			eachSession = listOfSessions[i]
			eachCycle = []
			levels = []
			timeD = []
			if len(eachSession) < 3 or (eachSession[-1][1] - eachSession[0][1]).total_seconds() < 5*60:
				continue

			span = (eachSession[-1][1] - eachSession[0][1]).total_seconds()
			bins = math.ceil(span/(60*10))		#no. of 10 min bins
			if span/60 > max_span:
				max_span = span/60
			for j in range(1,len(eachSession)-1):
				timeDelta = int((eachSession[j][1] - eachSession[0][1]).total_seconds()/60)
				bins = round(timeDelta/10)
				levelDelta = eachSession[0][0] - eachSession[j][0]
				eachCycle.append([levelDelta, bins])
				
			#trimmed = []
			prev = eachCycle[0]
			for k in range(1,len(eachCycle)):
				if prev[1] != eachCycle[k][1]:
					a = prev[1]
					b = prev[0]
					if day in [1,2,3,4,5]:
						w1[a].append(b)
					else:
						w2[a].append(b)
					#trimmed.append(eachCycle[k])
				prev = eachCycle[k]
			if day in [1,2,3,4,5]:
				w1[prev[1]].append(prev[0])
			else:
				w2[prev[1]].append(prev[0])
	print(max_span, len(w1), len(w2))
	return{'weekday':w1, 'weekend':w2}
			
def doInPool():
	pFile = pickle.load(open('/home/anudipa/Documents/Jouler_Extra/master_list_100.p','rb'), encoding='bytes')
	filtered = convert(pFile)
	#filtered = filtered[:100]
	print(filtered)
	pool = Pool(processes=8)
	res = pool.map(discharge, filtered)
	pool.close()
	pool.join()
	return res

print('*************')
#all_dump = doInPool()
#print(len(all_dump))
#weekdays = defaultdict(list)
#weekends = defaultdict(list)
#for i in range(len(all_dump)):
#	each_w1 = all_dump[i]['weekday']
	#print(len(each_w1[0]))
#	each_w2 = all_dump[i]['weekend']
#	for delta in each_w1.keys():
#		weekdays[delta] += (each_w1[delta])
#	for delta in each_w2.keys():
#		weekends[delta] += (each_w2[delta])
	#print(0, len(weekdays[0]))

allDays = discharge('cdfce167c0f7fbcdc14b841f0cf4cd6d7fe6d470')
weekdays = allDays['weekday']
fig, ax = plt.subplots()
data = []
N = 0
for delta in weekdays.keys():
	if delta > 130:
		break
	data.append(weekdays[delta])
	N += 1
print(N, len(data))
bp = ax.boxplot(data, sym='', vert=1, whis=[25,75])
ax.set_xticks(np.arange(0, N, 5))
ax.set_xticklabels(np.arange(0, N, 5))
ax.set_xlabel('dT (*10 mins)')
ax.set_ylabel('dL (from t=0)')
ax.set_title('Total level drop at dt on weekdays')
fig.show()

