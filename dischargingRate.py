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
from scipy.stats import itemfreq
import timeit
import testCases


path1a = '/home/anudipa/Documents/Jouler_Extra/discharge2/'
path1b = '/home/anudipa/pattern/discharge/'
path2a = '/home/anudipa/Documents/Jouler_Extra/master_list_100.p'
path2b = '/home/anudipa/pattern/master_list_100.p'

path_to_discharge = path1a
path_to_dev = path2a


def convert(data):
	data_type = type(data)
	if data_type == bytes : return data.decode()
	if data_type in (str,int,float): return data
	if data_type in (datetime.datetime,datetime.date): return data
	if data_type == dict: data = data.items()
	return data_type(map(convert, data))

def discharge(dev):
	file1 = path_to_discharge+dev+'.p'
	try:
		print('Starting', dev)
		tmp1 = pickle.load(open(file1, 'rb'), encoding='bytes')
		dDataSet = convert(tmp1)
		dData = dDataSet[dev]
	except Exception as e:
		print('Error at opening file', e)
		return

#dictionary to store discharging events: key ->  start_time, value-> list of discharging events
	dict_ = defaultdict(list)
	sortedDates = sorted(dData.keys())
#tracking start and end of charging sessions, first_ time a battery level seen at concerned discharge event
	track_first = dData[sortedDates[0]][0][0]
	track_last = track_first
	#last_ -> last event seen, first_ -> first time event for current level in the current session
	last_ = track_first
	first_ = track_first
	statusD = False			#True when discharging
#	flag = False
	issues = 0
	last_track_first = track_first
#	print('Total number of days:', len(sortedDates))
	for i in range(len(sortedDates)):
		d = sortedDates[i].weekday()
		listOfSessions = dData[sortedDates[i]]
#		if issues > 5:
#			break
		for j in range(len(listOfSessions)):
			eachSession = listOfSessions[j]
			#print('##',len(listOfSessions[j+1]))
			for k in range(len(eachSession)):
				flag = True
				#print('*****$#$#$#*****',j,listOfSessions[j][0][1], listOfSessions[j][-1][1], last_[1])
				#check if the last tracked timestamp is after the current timestamp. can happen if we predicted this to be an anomaly. so ignore.
				if last_[1] > eachSession[k][1] :
					if k == 0:
						print('Error, look into this!', last_, eachSession[k])
					else:
						print('Error, look into this!', last_, k, eachSession[k], k-1, eachSession[k-1])
					issues += 1
					continue
				#get at least 4 consecutive events starting now, this will help to determine fluctuation or anomalies in case of any status change
				events = eachSession[k:]
				if len(events) < 4:
					if j+1 < len(listOfSessions):
						events = events + listOfSessions[j+1]
					elif i+1 < len(sortedDates):
						p = 0
						while(len(events) < 4 and p < len(dData[sortedDates[i+1]])):
							events = events + dData[sortedDates[i+1]][p]
							p += 1
				if statusD and first_[0] != last_[0]:
					print('!!Not consistent', first_, last_, eachSession[k])
#				flag = checkFalsePos(events,statusD)
				#if two timestamps/events are too far apart or current level is higher than last recorded, then end earlier session and start new.
				if (eachSession[k][1] - last_[1]).total_seconds() > 15*60*60 and eachSession[k][0] > last_[0]:
					#statusD is true when discharging and false when charging
					if statusD:
						fillInTheBlank(dict_, last_, first_, track_first)
#						print('*', len(dict_[track_first[1]]), dict_[track_first[1]][0], dict_[track_first[1]][-1])
						track_last = last_
						track_first = eachSession[k]
						first_ = track_first
						last_ = track_first
						#check if this is start of another discharging session; if true add new session else change status
						flag = isThisTrueReading(eachSession[k], events, True)
						if flag:
							dict_[track_first[1]].append([track_first[1],track_first[0],0])
						else:
							statusD = False
				elif last_[0] > eachSession[k][0]:
					flag = isThisTrueReading(last_, events, True)
					if not statusD:
						#to check if event changed from charging to discharging	
						#flag = checkFalsePos(events, False)
						#flag = isThisTrue(last_, events, True)
						if flag:
							#print('*',last_[0], eachSession[k][0],first_[0],track_first[1])
							statusD = True
							if first_[0] == last_[0]:
								track_first = first_
							else:
								track_first = last_
							#track_first = eachSession[k]
							#print('***********')
							dict_[track_first[1]].append([track_first[1],track_first[0],0])
							first_ = eachSession[k]
					else:
#						sometimes there are wrong readings like a 0, these also fall under
#						unwanted fluctuations
						#check if the event is still continuing to be of type discharging
						#flag = checkFalsePos(events, False)
			
						if not flag:				#this is a fluctuation
							if dict_[track_first[1]][-1][1] < last_[0]:
								print('*',track_first[1], dict_[track_first[1]][-1], last_)
							#fillInTheBlank(dict_,last_, first_, track_first)
							#track_last = last_
							#first_ = events[1]
							if dict_[track_first[1]][-1][1] >= events[1][0]:
								#if still decreasing skip the wrong reading, and initialize first with next
								continue
							else:
								#end this level tracking
								fillInTheBlank(dict_,last_, first_, track_first)
								#first_ = events[1]
							#last_ = first_
							#track_first = events[1]
							#print('***#$@$@#@$#@$#$@#$@#$$@#**', events[:3], listOfSessions[j][0][1])
						else:					#this is legitimate drop
							if dict_[track_first[1]][-1][1] < last_[0]:
								print('**',track_first[1], dict_[track_first[1]][-1], last_)			
							fillInTheBlank(dict_, eachSession[k], first_, track_first)
							first_ = eachSession[k]
							track_last = eachSession[k]
				elif last_[0] < eachSession[k][0]:
					flag = isThisTrueReading(last_,events,False)			#checking if this is end of discharging session
					if statusD:
						if flag :
							#print('end session', last_, eachSession[k-1],k, len(eachSession))
							if dict_[track_first[1]][-1][1] < last_[0]:
								print('***',track_first[1], dict_[track_first[1]][-1], last_)
							#end the discharging session
							fillInTheBlank(dict_, last_, first_, track_first)
							#check if the discharging session just ended is too short(less than 10 mins), then ignore by deleting that key
							span = (dict_[track_first[1]][-1][0]-dict_[track_first[1]][0][0]).total_seconds()
							if span < 10*60:
								dict_.pop(track_first[1], None)
							#dict_[track_first[1]].append([last_[1], last_[0],0])
							track_last = last_
							#first_ = last_
							track_first = eachSession[k]
							first_ = track_first
							statusD = False
							
							

							
				if flag:
					last_ = eachSession[k]
				if last_track_first[1] != track_first[1]:
					if last_track_first[1] in dict_ and len(dict_[last_track_first[1]]) < 2:
						print('!!!!!!!!', last_track_first, track_first, events[:3])
				if first_[0] != last_[0] and statusD:
					print('*!*!*!*', first_, last_, eachSession[k])
					#else:
					#	print('*****', (dict_[last_track_first[1]][-1][0]-dict_[last_track_first[1]][0][0]).total_seconds()/60, dict_[last_track_first[1]][0][1] - dict_[last_track_first[1]][-1][1])
					last_track_first = track_first
#				print(last_)
	print('****', dev, len(dict_.keys()))
#	all_ = list(dict_.keys())
#	for i in range(len(all_)):
#		t = abs(dict_[all_[i]][0][0] - dict_[all_[i]][-1][0]).total_seconds()/60
#		print(i, t, dict_[all_[i]][0][0], dict_[all_[i]][0][1], dict_[all_[i]][-1][0], dict_[all_[i]][-1][1], len(dict_[all_[i]]))
#ToDo
#	cleanUp(dict_)
	test = testCases.testDischarging(dev, dict_)
	new_dict = {}
	if not test:
		d = cleanUp(dict_)
		new_dict[dev] = d
	else:
		new_dict[dev] = dict_
	return new_dict

def fillInTheBlank(dict_, last_event, first_event, start_session_event):
	diff = (last_event[1] - first_event[1]).total_seconds()
	drop = (first_event[0] - last_event[0])
	if diff == 0 :
		dict_[start_session_event[1]].append([first_event[1], first_event[0],0])
		return
	total_hours = int(diff/3600)
	
	rate_per_min = (drop/diff)*60
	if total_hours > 0:
		per_hour_drop = drop/total_hours
	else:
		per_hour_drop = 0
	t = 0
	level_now = last_event[0]
	time_now = last_event[1]
	if start_session_event[1] in dict_.keys():
		latest = max(dict_[start_session_event[1]], key=lambda x:x[0])
	else:
		latest = [start_session_event[1], start_session_event[0],0]
	while(t <= total_hours):
		dict_[start_session_event[1]].append([time_now, int(level_now), rate_per_min])
		t += 1
		time_now = last_event[1] - timedelta(hours=t)
		level_now += per_hour_drop
		if time_now < first_event[1] or (time_now >= first_event[1] and level_now > first_event[0]) or time_now < latest[0]:
			break
#returns true if session confirms to the status sent as argument or there is wrong reading
#since discharging rate can be anything, we will verify against charging rate.
#assumption it cannot be higher than 2.5 levels per min == 100 level in 45 mins approx
def isThisTrueReading(last, session, curr_status):
	if len(session) < 3:
#		print('Too small session length to decide')
		return True
	if curr_status:
		diff_mins = (session[1][1] - session[0][1]).total_seconds()/60.0
		if diff_mins == 0:
			charging_rate = -1
		else:
			charging_rate = (session[1][0] - session[0][0])/diff_mins
		if session[0][0] < session[1][0]:
			#is this a legitimate charging session or just anomalous reading
			if diff_mins == 0 or charging_rate > 3 or (last[0] >= session[1][0] and diff_mins < 5):
				#most likely session[0] is a wrong reading
				return False
#		if session[0][0] > session[1][0] and session[1][0] < session[2][0]:
#			if (session[1][1] - last[1]).total_seconds() < 10*60:
#				return False
			
	else:
	#check if this is end of a session or a wrong reading
		#is this the beginning of a new legitimate discharging session or just anomalous reading
		#things to check: is the charging time too short, is the charging rate too high
		#things to check: if the diff between next two event 0, 
		c_time = (session[0][1] -  last[1]).total_seconds()/60.0
		c_diff = session[0][0] - last[0]
		if c_time == 0:
			c_rate = 0
		else:
			c_rate = c_diff/c_time

		if c_diff < 2 or c_time < 10 or c_rate > 2.75:
			return False
		#elif last[0] == session[1][0] and (session[1][1] - last[1]).total_seconds < 1
		#if session[0][0] > session[1][0] and session[1][0] < session[2][0]:
		#	return False
	return True
			


#combine any sessions that got fragmented
def cleanUp(dict_):
	new_dict = defaultdict(list)
	sortedK = sorted(dict_.keys())
	corrected = 0
	#check 3 instances at a time, prev, curr and next. 1. check timespan and change of level
	#for curr. 2. check time interval and changeof battery level between prev and curr, and 
	#between curr and next. if time diff is too small or level difference is none or continues
	#to drop then merge prev. ###First try with 2 instances instead of 3 i.e. prev and curr
	prev = dict_[sortedK[0]]
	new_dict[sortedK[0]] = sorted(dict_[sortedK[0]], key=lambda x:x[0])
	track_k = sortedK[0]
	print('**Start cleanup: ')
	c = 0
	for i in range(1, len(sortedK)):
		curr = sorted(dict_[sortedK[i]], key = lambda x : x[0])
#		print(dict_[sortedK[i]][-1], dict_[sortedK[i]][0])
		#check if all the sessions are consistent, as in all are consistently decreasing
		slist = sorted(dict_[sortedK[i]], key=lambda x: x[0])
		l = len(slist)
	#	print('**', i, track_k, sortedK[i],len(curr))
		#if charging interval is short or charging diff is small, merge
		#options: 1. if the current session is short ( < 15 mins and < 3) , then it can be fluctuations seen during charging, discard current
		#options: 2. if the current session is okay (> 15 mins or > 3), then roll to the event whose battery level is equal or less than prev[-1]
		#		and merge
		interval = (curr[0][0] - new_dict[track_k][-1][0]).total_seconds()
		diff = curr[0][1] - new_dict[track_k][-1][1]
		curr_span = (curr[-1][0] - curr[0][0]).total_seconds()
		curr_diff = curr[0][1] - curr[-1][1]
#		if i==38 or curr[0][0] == datetime.datetime(2015, 3, 27, 17, 3, 8, 86300):
#			print('#1',curr[0], interval, diff, curr_span, curr_diff)
		
		if interval <= 15*60 or (diff < 3 and interval < 5*3600):
#			if i==38 or curr[0][0] == datetime.datetime(2015, 3, 27, 17, 3, 8, 86300):
#				print('#',curr[0], interval, diff, curr_span, curr_diff)
			#if interval between last session end and current session  is long
			if curr_span < 15*60 and curr_diff < 3 :
				#print('*Error corrected', curr[0], curr[-1])
				corrected += 1
				continue
			#start rolling
#			if i==38 or curr[0][0] == datetime.datetime(2015, 3, 27, 17, 3, 8, 86300):
#				print('#3', curr[0], interval, diff, curr_span, curr_diff)
			for j in range(len(curr)):
				if new_dict[track_k][-1][1] >= curr[j][1]:
					break
#			if i==38 or curr[0][0] == datetime.datetime(2015, 3, 27, 17, 3, 8, 86300):
#				print('#4', curr[0], interval, diff, curr_span, curr_diff, j, len(curr))
			if j < len(curr) -1:
				new_dict[track_k] += curr[j:]
				#print('Error corrected', new_dict[track_k][0], new_dict[track_k][-1])
				corrected += 1
				continue
		else:
			new_dict[sortedK[i]] = curr
			track_k = sortedK[i]
	keys = list(new_dict.keys())
	for k in keys:
		span = (new_dict[k][-1][0] - new_dict[k][0][0]).total_seconds()
		drop = (new_dict[k][0][1] - new_dict[k][1][1])
		if span <= 300 or (span <=600 and drop < 2):
			new_dict.pop(k,None)
			corrected += 1
	print('**Cleanup: Before:', len(dict_.keys()), 'After: ', len(new_dict.keys()), 'Corrected: ', corrected)
	return new_dict


#do in pool
def doInPool(x):
	pFile = pickle.load(open(path_to_dev,'rb'), encoding='bytes')
	filtered = convert(pFile)
	print(len(filtered))
	filtered = filtered[:x]
	#print(filtered)
	pool = Pool(processes = 4)
	res = pool.map(discharge, filtered)
	pool.close()
	pool.join()
	return res

def checkIf(session):
	slist = sorted(session, key=lambda x: x[0])
	for i in range(len(slist)-1):
		if slist[i][1] < slist[i+1][1]:
			print('not consistent', slist[i], slist[i+1])
			for j in range(len(slist)):
				print(slist[j][0], slist[j][1])
			return False
	return True

def testing():
	all_ = doInPool(1)
	dev = next(iter(all_[0]))
	dict_ = all_[0][dev]
	checkData(dict_)

def checkData(dict_):
	sortedK = sorted(dict_.keys())
	intervals = []
	span = []
	for d in range(len(sortedK)-1):
		#interval between sessions, and span of each session
		d1 = (dict_[sortedK[d+1]][0][0] - dict_[sortedK[d]][-1][0]).total_seconds()/60
		#intervals.append(round(d1,2))
		d2 = (dict_[sortedK[d]][-1][0] - dict_[sortedK[d]][0][0]).total_seconds()/60
		if d1 > 20000 or d2 > 20000:
			continue
		intervals.append(round(d1,2))
		span.append(round(d2,2))
		#print(d, d2, 'after', d1, '; drop =', dict_[sortedK[d]][0][1] - dict_[sortedK[d]][-1][1], 'inc by =', dict_[sortedK[d+1]][0][1] - dict_[sortedK[d]][-1][1])
	print(max(intervals), max(span))
	fig, ax = plt.subplots()
	ind = np.arange(len(intervals))
	ax.boxplot([intervals, span], sym='',whis=[25,75],patch_artist = True)
	fig.show()

def whatType(slist_):
	end = slist_[-1]
	start = slist_[0]
	span_mins = (end[0] -start[0]).total_seconds()/60
	if span_mins > 6*60:
		return -1
	pivot = start[0] + timedelta(minutes=int(span_mins/2))
	mid = min(slist_, key=lambda d: abs(d[0] - pivot))
        #calculate rate for first half and second half
	first_drop = start[1] - mid[1]
	second_drop = mid[1] - end[1]
	if (first_drop + second_drop) == 0 :
		return -1
#	first_diff_min = (mid[0] - start[0]).total_seconds()/60.0
#	second_diff_min = (end[0] - mid[0]).total_seconds()/60.0
	
#	first_rate = first_drop/first_diff_min
#	second_rate = second_drop/second_diff_min
	if first_drop >= 2*second_drop:
		return 0
	elif second_drop >= 2*first_drop:
		return 1
	else:
		return 2

def smallIntervals(dict_):
	l = len(dict_.keys())
	print('Possibly distinct discharging sessions', l)
	print('*************************************')
	sortedK = sorted(dict_.keys())
	for i in range(l):
		span = (dict_[sortedK[i]][-1][0] - dict_[sortedK[i]][0][0]).total_seconds()/60
		recorded = len(dict_[sortedK[i]])
		drop = (dict_[sortedK[i]][0][1] - dict_[sortedK[i]][-1][1])
		last_level = dict_[sortedK[i]][-1][1]
#		if i >0:
#			print(i, 'Starts after', (sortedK[i] - dict_[sortedK[i-1]][-1][0]).total_seconds()/60)
#		print(i, sortedK[i], span, drop, recorded, 'levels:', dict_[sortedK[i]][0][1], last_level)
		if drop == 0:
			print(i, sortedK[i], span, recorded, dict_[sortedK[i]])
			if i > 0:
				print(i, 'Starts after', (sortedK[i] - dict_[sortedK[i-1]][-1][0]).total_seconds()/60, dict_[sortedK[i-1]][0][1], sortedK[i-1], dict_[sortedK[i-1]][-1][1], dict_[sortedK[i-1]][-1][0])
			if i < l -1 :
				print(i, 'Next in ', abs(dict_[sortedK[i]][-1][0] - sortedK[i+1]).total_seconds()/60, dict_[sortedK[i+1]][0][1], sortedK[i+1], dict_[sortedK[i+1]][-1][0], dict_[sortedK[i+1]][-1][1])

def examineRates(all_):
	data = []
	for event in all_.keys():
		list_ = all_[event]
		slist_ = sorted(list_,key=lambda x: x[0])
		end_ = slist_[-1]
		start_ = slist_[0]
		type1 = whatType(slist_)
#		if not type1 == 1:
#			continue
		if abs(end_[0] - start_[0]).total_seconds() == 0 or start_[1] - end_[1] == 0:
			continue
		total_mins = (end_[0]-start_[0]).total_seconds()/60.0
		ideal_rate = ((start_[1] - end_[1])/(end_[0] - start_[0]).total_seconds())*60.0
		for i in range(1,len(slist_)):
			rate_now = slist_[i][2]
			if abs(rate_now - ideal_rate)/ideal_rate <= 0.1:
				#print(ideal_rate, rate_now)
				time_from_start = (slist_[i][0] - slist_[0][0]).total_seconds()/60.0
				frac = time_from_start/total_mins
				#print(ideal_rate, rate_now, frac)
				data.append(frac)
				break
	print(min(data))
	return data

def examineRates1(all_):
	data = []
	data1 = []
	count1 = 0
	count2 = 0
	for event in all_.keys():
		list_ = all_[event]
		slist_ = sorted(list_,key=lambda x: x[0], reverse=True)
		end_ = slist_[0]
		start_ = slist_[-1]
		if abs(end_[0] - start_[0]).total_seconds() == 0 or start_[1] - end_[1] == 0:
			continue
		total_mins = (end_[0]-start_[0]).total_seconds()/60.0
		ideal_rate = ((start_[1] - end_[1])/(end_[0] - start_[0]).total_seconds())*60.0
		flag = False
		t1 = t2 = slist_[0][0]
		r1 = r2 = slist_[0][2]
		for i in range(1, len(slist_)):
			rate_now = slist_[i][2]
#logic: see when rate_now is within required bounds, flag=1, t1=t2  and iterate; update t1 till flag is true
#default value of t1=t2=end of session
			if abs(rate_now - ideal_rate)/ideal_rate <= 0.1:
				if not flag:
					flag = True
					t2 = slist_[i][0]
					r2 = slist_[i][2]
				t1 = slist_[i][0]
				r1 = slist_[i][2]
				#print('Updating', t1, rate_now, ideal_rate)
			else:
				if flag:
					flag = False
					#t1 = slist_[0][0]
					#t2 = slist_[0][0]
					#print('Resetting', slist_[i][0], rate_now, ideal_rate)
		if (t2 - t1).total_seconds() > 30*60:
#			print(t1, r1, t2, r2, ideal_rate)
#			print('***', start_[0], start_[1], end_[0], end_[1], total_mins)
			t1_from_start = (t1 - start_[0]).total_seconds()/60
			t2_from_start = (t2 - start_[0]).total_seconds()/60
			frac1 = t1_from_start/total_mins
			frac2 = t2_from_start/total_mins
			if frac2 > 0.8:
				data.append(frac1)
				data1.append(t1_from_start)
				count1 += 1
#				print(t1, r1, t2, r2, frac1)
#				print('***', start_[0], start_[1], end_[0], end_[1], total_mins)
			else:
				count2 += 1

	print(count1, count2)
	return data1	

def getIdealRate(all_):
	dict_ = defaultdict(list)
	for event in all_.keys():
		list_ = all_[event]
		slist_ = sorted(list_, key=lambda x: x[0])
		start_level = slist_[0][1]
		if (slist_[-1][0] - slist_[0][0]).total_seconds() < 10*60 or slist_[-1][1] == slist_[0][1]:
			continue
		total_mins  = (slist_[-1][0] - slist_[0][0]).total_seconds()/60
		ideal_rate = (slist_[0][1] - slist_[-1][1])/total_mins
		dict_[start_level].append(ideal_rate)
	return dict_
	

#forecast
def forecast(all_):
#1. get list of pairs of <mean discharge rate, starting battery level>
#1a. calculate mean ealiest time, and mean discharge rate
#2. compute end of session
#3. plot difference between predicted and actual end
	data = []
	for i in range(len(all_dump)):
		all_ = all_dump[i]
		dev = next(iter(all_))
		each = examineRates1(all_[dev])
		mins = np.mean(each)
		print('****',mins)
		rate_level = getIdealRate(all_[dev])
#forecast for last 30 days
		sortedD = sorted(all_[dev].keys(), reverse=True)
		i = 0
		res = []
		count = 0
		while(count < 30):
			list_ = all_[dev][sortedD[i]]
			slist_ = sorted(list_, key=lambda x: x[0])
			start_ = slist_[0]
			span = (slist_[-1][0] - slist_[0][0]).total_seconds()/60
			level = (slist_[0][1] - slist_[-1][1])
			if span < 30 or level <= 0 or span > 4*24*60:
				i += 1
				#print('Skipping #',i-1,slist_[0][0], slist_[0][1], slist_[-1][0], slist_[-1][1]) 
				continue
#get the ideal rate for this session based on the start battery level
			ideal_rate = np.mean(rate_level[start_[1]])
			pivot = start_[0] + timedelta(minutes=(mins))
			flagged = min(slist_, key=lambda d: abs(d[0] - pivot))
			rate_ = flagged[2]
			level = flagged[1]
			if rate_ == 0:
				#print('Skipping', flagged)
				i += 1
				continue
#			if rate_ < ideal_rate:
#				rate_ = ideal_rate
			mins_pred = int(level/rate_)
			end_pred = flagged[0] + timedelta(minutes=mins_pred)
			end_true = slist_[-1][0]
#find the most realistic time when this session would have ended in battery level zero
			avg_rate = (level/span)
			if not slist_[-1][1] == 0:
				more_minutes = (slist_[-1][1])/avg_rate
				new_end_true = end_true + timedelta(minutes=more_minutes)
			else:
				new_end_true = end_true
			if abs(end_pred - end_true).total_seconds() > 2* 24* 3600:
				i += 1
				continue
#			print('#',i,start_[0], start_[1], flagged[0], flagged[1], end_pred,'-->', end_true, slist_[-1][1], new_end_true, ' diff', abs((end_pred - new_end_true).total_seconds()/60))
			count += 1
			res.append((end_pred - new_end_true).total_seconds()/60)
			i += 1
		print(res)
		data.append(res)
	return data

##########PLOTTING GRAPHS FUNCTIONS#########
def plotForecast(all_dump):
	fig, ax = plt.subplots()
	data = []
	

def plotEarly(all_dump):
	fig, ax = plt.subplots()
	data = []
	for i in range(len(all_dump)):
		all_ = all_dump[i]
		dev = next(iter(all_))
		each = examineRates1(all_[dev])
#		print(each)
		if len(each) > 3:
			data.append(each)
		else:
			print(i, 'Not working')
	if len(data) == 0:
		return	
	bp = ax.boxplot(data, sym='',whis=[25,75],patch_artist = True)
	for box in bp['boxes']:
		box.set(color='#FFFFFF', linewidth=1)
		box.set(facecolor='#696969')
	for median in bp['medians']:
		median.set(color= '#FF0000', linewidth=4)
	ax.tick_params(labelbottom='off')
	ax.set_ylabel('delta T from start/total time span')
	ax.set_xlabel('Users')
#	ax.set_title('For sessions > 6 hrs and Type 1, how early rate comes close to avg discharge rate')
	fig.show()


def plotTypesOfSession(all_dump):
	fig, ax = plt.subplots()
	data = []
	for i in range(len(all_dump)):
		dev = next(iter(all_dump[i]))
		all_ = all_dump[i][dev]
		print(dev)
		highC = lowC = count = 0
		weird = []
		for event in all_.keys():
			list_ = all_[event]
			#print(len(list_), list_[0])
			slist_ = sorted(list_, key=lambda x: x[0])
			start = slist_[0]
			end = slist_[-1]
			span_mins = (end[0] - start[0]).total_seconds()/60.0
#condition for filtering out sessions, right now any session longer than 6 hours
			if span_mins < 3*60:
				if span_mins < 5:
#					print('Something is wrong! too short', slist_)
					weird.append([span_mins, start[1]-end[1]])
				continue
			count += 1
#get mid point for the session, and calculate rate for first and second half
			pivot = start[0] + timedelta(minutes=(span_mins/2))
			mid = min(slist_, key=lambda d: abs(d[0] - pivot))
			first_drop = start[1] - mid[1]
			second_drop = mid[1] - end[1]
			first_diff_min = (mid[0] - start[0]).total_seconds()/60
			second_diff_min = (end[0] - mid[0]).total_seconds()/60
			#first_rate = first_drop/first_diff
			#second_rate = seconds_drop/second_rate
			if first_drop+second_drop == 0 or first_diff_min==0:
#				print('Ignore this session',first_drop, second_drop, first_diff_min)
				continue
#			print(first_drop, first_diff_min, second_drop, second_diff_min)
			if first_drop >= 2*second_drop:
				highC += 1
			elif second_drop >= 2*first_drop:
				lowC += 1
		print('Breaking it down: ', count, (highC/count)*100, (lowC/count)*100, ((count -highC-lowC)/count)*100)
		print('**', len(weird), weird)
		data.append([(highC/count)*100, (lowC/count)*100, ((count -highC-lowC)/count)*100])
	for i in range(len(data)):
		b0 = ax.bar(i,data[i][0],width=0.5, color='g')
		b1 = ax.bar(i, data[i][1], width=0.5,color='r',bottom = data[i][0])
		b2 = ax.bar(i, data[i][2], width=0.5,color='b',bottom=data[i][0]+data[i][1])
	ax.legend(labels=['High drain in 1st half','High drain in 2nd half','Other'],loc='upper left')
	ax.set_title('For discharging sessions lesser than 12 hours')
	fig.show()

#Number of charging events per day for each device
def chargingEvent(dict_, dev):
	results = []
	sortedD = sorted(dict_.keys())
	curr = sortedD[0].date()
	count = 0
	session = [1]
	days = [curr]
	for i in range(1,len(sortedD)):
		if sortedD[i].date() != curr:
			curr = sortedD[i].date()
			session.append(1)
			days.append(curr)
		else:
			session[-1] += 1

	f = itemfreq(session)
	most_freq = max(itemfreq(session), key= lambda x: x[1])		
	count012 = sum([x[1] for x in f if x[0] in [0,1,2]])
	count3 = sum([x[1] for x in f if x[0] > 3])
	#print([x[1] for x in f if x[0] > 3])
	if count012 > len(session)/2:
		print(dev, 'Category 1: 0 to 2 events per day', most_freq, count012, len(session))
	elif count3 > count012:
		print(dev, 'Category 2: 3 or more events per day', most_freq, count3, len(session))
	else:
		print(dev, 'Uncategorized', f)
	startLevel(dict_, dev, session, days)
	return

#how length of session adn ending battery level varies if we know whatbattery level the session start
def startLevelTest(dict_, dev, test1, test2):
	levels = [] 			#[start, end]
	length = []
	sortedD = sorted(dict_.keys())
#	for i in range(len(sortedD)):
#		#get start and end battery level
#		levels.append([dict_[sortedD[i]][0][1], dict_[sortedD[i]][-1][1]])
#		length.append((dict_[sortedD[i]][-1][0] - dict_[sortedD[i]][0][0]).total_seconds()/60.0)
#	high = [x[1] for x in levels if x[0] > 80]
#	lg = [length[i] for i in range(len(levels)) if levels[i][0] > 80]
#	print(lg)
	dayToTrack = []
	lg = []
	j = 0
	for i in range(len(test2)):
		if test1[i] > 2:
			continue
		dayToTrack.append(test2[i])
		if j > len(sortedD):
			break
		while(sortedD[j].date() <= dayToTrack[-1]):
			if sortedD[j].date() < dayToTrack[-1]:
				#print(sortedD[j].date(), dayToTrack[-1])
				j += 1
				continue
			if dict_[sortedD[j]][-1][1] > 50:
				print(dict_[sortedD[j]][-1], dayToTrack[-1])
				lg.append((dict_[sortedD[i]][-1][0] - dict_[sortedD[i]][0][0]).total_seconds()/60.0)
			#if dict_[sortedD[j]][0][1] > 80:
			levels.append(dict_[sortedD[j]][-1][1])
			j += 1

	#fig, ax = plt.subplots()
	#ax.hist(lg)
	too_low = [lg[i] for i in range(len(lg)) if lg[i] < 30] 
	print(len(dayToTrack),len(lg), too_low)
	print(itemfreq(levels)[:5])
	#fig.show()


def funForAll(all_):
	print('Start checking********************************')
	for i in range(len(all_)):
		dev = next(iter(all_[i]))
		dict_ = all_[i][dev]
		#chargingEvent(dict_, dev)
		print(dev, len(dict_.keys()))
		st = testCases.testDischarging(dev, dict_)


#Beginning
#start_time = timeit.default_timer()
#print('*************')
#all_dump = doInPool()
#funForAll(all_dump)
#print('Number of devices',len(all_dump))
#all_dump = discharge('0c037a6e55da4e024d9e64d97114c642695c5434')

#elapsed = timeit.default_timer() - start_time
#print('Halfway through: ', elapsed)
#find rate of discharge for first and second half of discharging session
#plotEarly(all_dump)
#plotTypesOfSession(all_dump)
#forecast(all_dump)
#sys.exit()
#for i in range(len(all_dump)):
#	dev = next(iter(all_dump[i]))
#	dict_ = all_dump[i][dev]
#	print(dev)
#	smallIntervals(dict_)
#smallIntervals(all_dump[next(iter(all_dump))])
#elapsed = timeit.default_timer() - start_time
#print('Completed in', elapsed)

