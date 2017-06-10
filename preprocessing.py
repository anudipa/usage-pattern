#!/usr/bin/env python2
import os
import sys
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import json
import operator
from datetime import *
from collections import defaultdict
import pickle
from multiprocessing import Process, Manager, current_process, Pool

def convert(data):
	data_type = type(data)
	if data_type == bytes : return data.decode()
	if data_type in (str,int,float,bool): return data
	if data_type in (datetime,datetime.date): return data
	if data_type == dict: data = data.items()
	return data_type(map(convert, data))


def loadAll(root, func=1, num=-1):
	devices = []
	for f in os.listdir(root):
		name = os.path.join(root, f)
		if os.path.isfile(name):
			devices.append(name)
	if -1 < num < len(devices):
		inputD = devices[:num]
	else:
		inputD = devices
	if len(inputD) > 0:
		pool = Pool(processes=8)
		if func == 1:
			res = pool.map_async(loadFirstRound, inputD)
		else:
			res = pool.map_async(loadSecondRound, inputD) 
		pool.close()
		pool.join()

	print('About to pickle results', len(res))
#	for i in range(len(res)):
#		for dev in res[i]:
#			if (res[i][dev].keys()) > 50:
#				name = os.path.join(root, 'final/discharge/'+dev+'.p')
#				pFile = open(name, 'wb')
#				pickle.dump(res[i],pFile)
#				pFile.close()

	print('End of pickling!!')
	return True

#first round of preprossing, get all discharging data filtered based on Status value
def loadFirstRound(path):
	try:
		device = (path.split('/')[-1]).split('.')[0]
		print('Start pickling', device)
		tmp_ = pickle.load(open(path, 'rb'), encoding='bytes')
		dict_ = convert(tmp_)
		print('keys:', len(dict_.keys()))
	except Exception as e:
		print(path, e)
		d = defaultdict()
		d[device] = {}
		return d
	#now process each key-value pair and select and store all discharge values
	all_ = defaultdict(list)
	#sort all timestamp
	sortedK = sorted(dict_.keys())
	#flag to point if current session is discharging or not
	isDischarging = False
	lastLevel = -1
	level = -1
	current = sortedK[0]
	for i in range(len(sortedK)):
		timestamp = sortedK[i]
		if 'BatteryProperties' not in dict_[timestamp].keys():
			continue
		level = dict_[timestamp]['BatteryProperties']['Level']
		status = dict_[timestamp]['BatteryProperties']['Status']
		status_data_type = type(status)
		try:
			if (status_data_type == str and status == 'Discharging') or (status_data_type == int and status > 2):
				if not isDischarging:
					current = timestamp
				all_[current].append([timestamp, level])
				isDischarging = True

			else:
				isDischarging = False
		except Exception as e:
			print('Error while reading levels', timestamp, e)
	result = defaultdict()
	result[device] = all_
	return result

def loadSecondRound(path):
	try:
		device = (path.split('/')[-1]).split('.')[0]
		tmp = pickle.load(open(path, 'rb'))
		dData = tmp[device]
	except Exception as e:
		print(path, e)
		d = defaultdict()
		return d
	
#dictionary to store discharging events: key ->  start_time, value-> list of discharging events belonging to that session
	dict_ = defaultdict(list)
	sortedD = sorted(dData.keys())

	statusD = False                 #True when discharging
	start_ = sortedD[0]
	for i in range(len(sortedD)):
		#sort all events timestamped in ascending order
		readings = dData[sortedD[i]]
		sortedT = sorted(readings, key=lambda x : x[0])
		if i == 0:
			last_ = sortedT[0]
#		print(sortedT)
#		break
#		readings = dData[sortedD[i]]
		for j in range(len(sortedT)):
			if  last_[1] < sortedT[j][1]:
				flag = isThisTrue(last_, sortedT[j:])
				if not flag:
					continue
				else:
					start_ = sortedT[j][0]
			if len(dict_[start_]) > 0 and (sortedT[j][0] - dict_[start_][-1][0]).total_seconds() > 3600:
				fillInTheBlanks(dict_, start_, sortedT[j])
			else:
				dict_[start_].append([sortedT[j][0], sortedT[j][1], 0.0])
			last_ = sortedT[j]
#		if i > 120:
#			break
#	err = testDischarge(dict_,device)
	new_dict = cleanUp(dict_)
#	err = testDischarge(new_dict, device)
	print(device, 'before', len(dict_.keys()), 'after', len(new_dict.keys()))
#now add number of days and wrap the dictionary of discharging sessions
	dates_ = []
	for k in new_dict.keys():
		if k.date() not in dates_:
			dates_.append(k.date())
	results = defaultdict(defaultdict)
	results[device]['days'] = dates_
	results[device]['sessions'] = new_dict
	return results


#return false if fluctuation, true if start of new session
def isThisTrue(last_, readings):
	#interval is short: if last_ < readings[0] but last_ >= readings[1]
	interval = (readings[0][0] - last_[0]).total_seconds()/60
	diff = readings[0][1] - last_[1]
	rate_ = round(diff/interval, 3)
	if (interval <= 120 and diff <= 2) or rate_ > 4:
		#either charging session is too short to be effective or increase in level is too low for it to be a charging session
		return False
	next_ = readings[0]
	j = 10000
	for i in range(1,len(readings)):
		if last_[1] >= readings[i][1]:
			next_ = readings[i]
			j = i
#			print('Inside func', j, last_, readings[0],next_)
			break
	#print('Inside func', j, last_, next_)
	if j < 3 and (next_[0]- readings[0][0]).total_seconds() < 5*60 and diff >= 5:
		#if diff > 5:
		return False
	return True				

def fillInTheBlanks(dict_, start_, end_):
	#pseudocode:
	#calculate number of hours(diff) and level difference(drop) between dict_[start][-1] and end_
	#while t is less than end_, add new [t, l, rate] for every hour increase starting from dict_[start][-1]
	diff = int((end_[0] - dict_[start_][-1][0]).total_seconds()/3600)
	drop = dict_[start_][-1][1] - end_[1]
	if diff <= 1:
		return
	dec_per_hour = int(drop/diff)
	hrs = 1
#	print('*', dict_[start_][-1], diff)
	while True:
		time_now = dict_[start_][-1][0] + timedelta(hours=1)
		level_now = dict_[start_][-1][1] - dec_per_hour
		if time_now >= end_[0] or level_now < end_[1] or hrs > diff:
			break
		dict_[start_].append([time_now, level_now, 0.0])
#		print('**',dict_[start_][-1])
		hrs += 1
	dict_[start_].append([end_[0], end_[1], 0.0])
#	print('***', dict_[start_][-1])
	return


def testDischarge(dict_, dev=None):
	err = 0
	err1 = 0
	sortedK = sorted(dict_.keys())
	for i in range(len(sortedK)):
		events = dict_[sortedK[i]]
		if i > 0:
			if dict_[sortedK[i-1]][-1][0] >= events[0][0]:
				print('Wrong session start & end time!!',i, dict_[sortedK[i-1]][-1], events[0])
				err += 1 
			elif dict_[sortedK[i-1]][-1][1] >= events[0][1]:
				print('Wrong session start & end levels', i, dict_[sortedK[i-1]][-1], events[0])
				err += 1
			if 0 <=(events[0][1] - dict_[sortedK[i-1]][-1][1])<= 2 and (events[0][0] - dict_[sortedK[i-1]][-1][0]).total_seconds() <= 5*3600:
				print('Inconsequential charge session', i, dict_[sortedK[i-1]][-1], events[0])
				err1 += 1
			elif (events[0][0] - dict_[sortedK[i-1]][-1][0]).total_seconds() <= 3*60:
				print('Too short charging session', i, dict_[sortedK[i-1]][-1], events[0])
				err1 += 1
			if (events[-1][0] - events[0][0]).total_seconds() < 10*60:
				print('Too short discharging session', len(events), round((events[-1][0] - events[0][0]).total_seconds()/60,2), (events[0][1] -events[-1][1]))
				err1 += 1
		for j in range(1,len(events)):
			if events[j-1][0] > events[j][0]:
				print('Wrong order of events by time!', events[j-1], events[j])
				err += 1
			if  events[j-1][1] < events[j][1]:
				print('Wrong order of events by level!', events[j-1], events[j])
				err += 1
			if (events[j][0] - events[j-1][0]).total_seconds() > 2*3600:
				print('Too long interval between events! do something!', events[j-1], events[j])
				err1 += 1
	print(dev, 'Errors', err, 'May consider correcting', err1)
	return err

def tempTest(list_):
	for k in range(len(list_)-1):
		if list_[k][1] < list_[k+1][1]:
			print('TT',k,list_[k], list_[k+1])


def cleanUp(dict_):
	sortedD = sorted(dict_.keys())
	new_dict = defaultdict(list)
	new_dict[sortedD[0]] = sorted(dict_[sortedD[0]], key=lambda x: x[0])
	added_ = sortedD[0]
	c = 0
	for i in range(1,len(sortedD)):
	#if battery level of 1st event of i-th event is less than the last event of i-1th event, then merge
	#dont do anything if difference is above 12 hours
	#also merge session if charging increased level by 2 or less.
		slist = sorted(dict_[sortedD[i]], key=lambda x: x[0])
#		for k in range(len(slist)-1):
#			if slist[k][1] < slist[k+1][1]:
#				print('*******', slist[k], slist[k+1], sortedD[i])
		event = slist[0]
		last_event = new_dict[added_][-1]
		interval = (event[0] - last_event[0]).total_seconds()
		curr_span = (slist[-1][0] - slist[0][0]).total_seconds()
		last_span = (new_dict[added_][-1][0] - new_dict[added_][0][0]).total_seconds()
#		if curr_span <= 10*60:
#			print('@@', added_, sortedD[i], last_event, int(interval/60),slist[0], slist[-1], int(last_span))
		if event[1] <= last_event[1] and interval < 12*3600:
			if (event[0] - last_event[0]).total_seconds() >= 2*3600:
				fillInTheBlanks(new_dict, added_, slist[0])
				new_dict[added_] += slist[1:]
			else:	
				new_dict[added_] += slist
			#tempTest('!@!',new_dict[added_], event, last_event)
			continue
		elif -2<(event[1] - last_event[1])<=2 and interval < 12*3600:
			for j in range(len(slist)):
				if last_event[1] >= slist[j][1]:
					break
			if j < len(slist)-1:
				if (slist[j][0] - last_event[0]).total_seconds() >= 2*3600:
					fillInTheBlanks(new_dict, added_, slist[j])
					new_dict[added_] += slist[j+1:]
				else:
					new_dict[added_] += slist[j:]
				#tempTest(new_dict[added_], event, last_event)
				continue
		elif (curr_span <= 10*60 or last_span <= 10*60) and interval < 12*3600:		#if last recorded session or current session is too short, check if they can be merged
			flag = False
			for j in range(len(slist)):
				if last_event[1] >= slist[j][1]:
					flag = True
					#print('**',j, int((slist[j][0] - slist[0][0]).total_seconds()/60))
					break
			if not flag and last_span <= 10*60 and (new_dict[added_][0][1] - new_dict[added_][-1][1]) <= 5:
				#print(last_event, event, len(new_dict[added_]), len(slist))
				new_dict.pop(added_, None)
			elif flag and j<= 3 and (slist[j][0] - slist[0][0]).total_seconds() <= 5*60:
				if (slist[j][0] - last_event[0]).total_seconds() >= 2*3600:
					fillInTheBlanks(new_dict, added_, slist[j])
					new_dict[added_] += slist[j+1:]
				else:
					new_dict[added_] += slist[j:]
				#print('**',c)
				c += 1
				#tempTest(new_dict[added_], event, last_event)
				continue
		if last_span <= 10*60 and (event[1] > last_event[1]):
#				print('!!', interval, last_event, len(slist))
				new_dict.pop(added_, None)
		#temporary testing
		#for j in range(1,len(new_dict[added_])):
		#	if new_dict[added_][j][1] > new_dict[added_][j-1][1]:
		#		print(new_dict[added_][j-1:])
		#		print(sortedD[i], sortedD[i-1])
		#		break
		new_dict[sortedD[i]] = slist
		added_ = sortedD[i]
#	slist = sorted(new_dict.keys())
#	for i in range(1,len(slist)):
#		if (new_dict[slist[i]][-1][0] - new_dict[slist[i]][0][0]).total_seconds() < 10*60:
#			ct1 = new_dict[slist[i]][0][1] - new_dict[slist[i]][-1][1]
#			ct2 = new_dict[slist[i-1]][0][1] - new_dict[slist[i-1]][-1][1]
#			cd2 = round((new_dict[slist[i-1]][-1][0] - new_dict[slist[i-1]][0][0]).total_seconds()/60, 2)
#			diff = new_dict[slist[i]][0][1] - new_dict[slist[i-1]][-1][1]
#			print('##', slist[i], ct1, ct2, cd2, ':', diff)
	return new_dict

if __name__ == "__main__":
#	root = '/home/anudipa/Documents/Jouler_Extra/pickles/Battery/'		
	root = '/home/anudipa/Documents/Jouler_Extra/final/discharge/'
	devices = []
	for f in os.listdir(root):
		name = os.path.join(root, f)
		if os.path.isfile(name):
			devices.append(name)
	inputD = devices
	print(inputD)
	pool = Pool(processes=8)
	res = pool.map(loadSecondRound, inputD)
	pool.close()
	pool.join()
	print('About to pickle results', len(res))
	c = 0
	for i in range(len(res)):
		dev = next(iter(res[i]))
		print(dev,len(res[i][dev].keys()))
#check if number of days is greater than 180, if true then pickle
		days = len(res[i][dev]['days'])
		sessions = len(res[i][dev]['sessions'].keys())
		if days >= 180 and sessions >= 100:
			name = os.path.join('/home/anudipa/Documents/Jouler_Extra/final/shortlisted/',dev+'.p')
			print(name, days, sessions)
			pFile = open(name, 'wb')
			pickle.dump(res[i],pFile)
			pFile.close()
			c += 1
	print('End of pickling!!', c)


