#!/usr/bin/env python2
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import *
from collections import defaultdict, Counter
import pickle
from multiprocessing import Process, Manager, current_process, Pool


def convert(data):
	data_type = type(data)
	if data_type == bytes: return data.decode()
	if data_type in (str, int, float, bool): return data
	if data_type in (datetime, datetime.date): return data
	if data_type == dict: data = data.items()
	return data_type(map(convert, data))

#load from shortlisted pickles, process in parallel, call checkFucn
def loadAll(root):
	devices = []
	for f in os.listdir(root):
		name = os.path.join(root, f)
		if os.path.isfile(name):
			devices.append(name)
	
	if len(devices) > 0:
		pool = Pool(processes=4)
		res= pool.map_async(checkFunc, devices)
		pool.close()
		pool.join()



#find number of charging sessions per day, if avg charging session is 3 or less
#or if avg level at start of charge is less than 20, then it is conservative
#print avg number of sessions, avg level at start, conservative or not

def checkFunc(path):
	try:
		device = (path.split('/')[-1]).split('.')[0]
		#print('Start loading', device)
		tmp_ = pickle.load(open(path, 'rb'), encoding='bytes')
		dData = tmp_[device]
		#print('keys:', len(dData.keys()), dData.keys(), device)
	except Exception as e:
		print(path, e)
		d = defaultdict()
		#d[device] = {}
		return 'error'
	#aim: to gather avg charging sessions per day & avg start level for charging session
	#	a. store number of charging sessions for each day [ 1 D array]
	#	b. store start level for every charging session per day [n D list]
	dict_ = defaultdict(list)
	num_of_sessions = list()
	num_of_sessions.append(0)
	start_level = list()
	count = 0
	#sort all days
	sortedD = sorted(dData['sessions'].keys())
	new_date = sortedD[0].date()
	last_entry = sorted(dData['sessions'][sortedD[0]], key=lambda x : x[0])[-1]
	prev_date =  sortedD[0]
	#print('Last entry', last_entry)
	dict_[new_date] = list()
	#print(new_date)
	#start_ = sortedD[0]
	for i in range(1,len(sortedD)):
		readings = dData['sessions'][sortedD[i]]
		sortedT = sorted(readings, key=lambda x : x[0])
#		if new_date not equals sortedD[i].date():
			#check if we have to update prev day's counts
			#sort all events timestamped in ascending order
		#print(new_date, last_entry, sortedT[0])
		if last_entry[0].date() == new_date and last_entry[1] < sortedT[0][1]:
			num_of_sessions[-1] += 1
			start_level.append(last_entry[1])
			dict_[new_date].append(last_entry[1])
			last_entry = sortedT[-1]
			#print('Updating: ', sortedD[i], num_of_sessions[-1], start_level[-1], last_entry[0])
		#now check of new_date has to be updated
		if new_date != sortedD[i].date():
			#print('*****', new_date, sortedD[i].date())
			new_date = sortedD[i].date()
			if last_entry[0].date() == new_date:
				num_of_sessions.append(1)
				start_level.append(last_entry[1])
			else:
				num_of_sessions.append(0)
				#if sortedT[-1][0].date() != new_date:
				#	print('Zero charging days', new_date, sortedT[0], sortedT[-1])
			last_entry = sortedT[-1]
			first_entry = sortedT[0]
			#num_of_sessions.append(0)
			dict_[new_date] = list()
		
		

	#print avg num of charging sessions/ avg start level at charge
#	print('Average start level', np.mean(start_level), len(start_level))
#	print('Average number of sessions',np.mean(num_of_sessions), sum(num_of_sessions))
#	print('*******************')
#	print(num_of_sessions)
#	print('*******************')
#	print(start_level)
	avg_sessions = np.mean(num_of_sessions)
	avg_level = np.mean(start_level)
	freq_level = Counter(start_level).most_common(3)
	if avg_sessions <= 3 or avg_level <= 25:
		print(device, 'is conservative user: #of sessions per day=', avg_sessions, ' avg start level=', avg_level, ' ::: ', freq_level, len(start_level),'  #of days=', len(dict_.keys()))
	else:
		print(device, 'is not conservative user: #of sessions per day=', avg_sessions, ' avg start level=', avg_level, ' ::: ', freq_level, len(start_level),'  #of days=', len(dict_.keys()))
	return device


if __name__ == "__main__":
	root = '/home/anudipa/pattern/git_scripts/usage-pattern/data/shortlisted'
	devices = []
	#checkFunc('/home/anudipa/pattern/git_scripts/usage-pattern/data/shortlisted/4a19576c087c9492ad6584f6885cebbd604b2c94.p')
	loadAll(root)
