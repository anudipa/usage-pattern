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

def convert(data):
	data_type = type(data)
	if data_type == bytes : return data.decode()
	if data_type in (str,int,float): return data
	if data_type in (datetime.datetime,datetime.date): return data
	if data_type == dict: data = data.items()
	return data_type(map(convert, data))

def discharge(dev):
	file1 = '/home/anudipa/Documents/Jouler_Extra/discharge2/'+dev+'.p'
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
				flag = False
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
							track_first = last_
							#track_first = eachSession[k]
							dict_[track_first[1]].append([track_first[1],track_first[0],0])
							#first_ = track_first
							#if first_[0] < eachSession[k][0]:
							#print('start discharge', track_first)
							#fillInTheBlank(dict_, eachSession[k], first_, track_first)
							first_ = eachSession[k]
					else:
#						sometimes there are wrong readings like a 0, these also fall under
#						unwanted fluctuations
						#check if the event is still continuing to be of type discharging
						#flag = checkFalsePos(events, False)
			
						if not flag:				#this is a fluctuation
							fillInTheBlank(dict_,last_, first_, track_first)
							track_last = last_
							first_ = events[1]
							last_ = first_
							#track_first = events[1]
							#print('***#$@$@#@$#@$#$@#$@#$$@#**', events[:3], listOfSessions[j][0][1])
						else:					#this is legitimate drop			
							fillInTheBlank(dict_, eachSession[k], first_, track_first)
							first_ = eachSession[k]
							track_last = eachSession[k]
				elif last_[0] < eachSession[k][0]:
					flag = isThisTrueReading(last_,events,False)			#checking if this is charging session
					if statusD:
						if flag :
							#print('end session', last_, eachSession[k-1],k, len(eachSession))
							#end the discharging session
							fillInTheBlank(dict_, last_, first_, track_first)
							#dict_[track_first[1]].append([last_[1], last_[0],0])
							track_last = last_
							#first_ = last_
							track_first = eachSession[k]
							first_ = track_first
							statusD = False
							
				if flag:
					last_ = eachSession[k]
				flag = False
#				print(last_)
	print('****', len(dict_.keys()))
#	all_ = list(dict_.keys())
#	for i in range(len(all_)):
#		t = abs(dict_[all_[i]][0][0] - dict_[all_[i]][-1][0]).total_seconds()/60
#		print(i, t, dict_[all_[i]][0][0], dict_[all_[i]][0][1], dict_[all_[i]][-1][0], dict_[all_[i]][-1][1], len(dict_[all_[i]]))
#ToDo
	cleanUp(dict_)
	new_dict = {}
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
		print('Too small session length to decide')
		return True
	diff_mins = (session[1][1] - session[0][1]).total_seconds()/60.0
	if diff_mins == 0:
		charging_rate = -1
	else:
		charging_rate = (session[1][0] - session[0][0])/diff_mins
	if curr_status:
		if session[0][0] < session[1][0]:
			#is this a legitimate charging session or just anomalous reading
			if diff_mins == 0 or charging_rate > 3 or (last[0] >= session[1][0] and diff_mins < 5):
				#most likely session[0] is a wrong reading
				return False
			
	else:
		if session[0][0] > session[1][0]:
			#is this a legitimate discharging session or just anomalous reading
			if diff_mins == 0 or (session[0][1] - last[1]).total_seconds() == 0 or (session[0][0] - last[0])/((session[0][1] - last[1]).total_seconds()/60) > 3:
				return False
	return True
			


#flag =  False if this a charging event, True if this is a discharging event, returns true always if
#fluctuation is detected and false if it is truly a change in charging status
def checkFalsePos(eachSession, flag):
# check 3 consecutive events 
	flaw = 0		#count is increased if there is discripency
	if len(eachSession) < 4:
#		print('No more sessions to add', eachSession)
		return False
	for i in [0,1,2]:
#if the decrease or increase is seen in consecutive events increase flaw else decrease. also if two events
#are further than 1 hour apart increase flaw and break, its too far apart to decide fluctuation
#if flaw if > 1 then its not a fluctuation else it is a fluctuation
		if (eachSession[i+1][1] - eachSession[i][1]).total_seconds() >= 60*60: #or abs(eachSession[i][0] - eachSession[i+1][0]) >= 10:
			flaw += 2
			break 
		if not flag:			#check if it is consistently deceasing
			if eachSession[i][0] >= eachSession[i+1][0]:
				flaw += 1
			else:
				flaw -= 1
				break	
		else:				#check if it is consistently increasing
			if eachSession[i][0] <= eachSession[i+1][0]:
				flaw += 1
			else:
				flaw -=1
				break
	if flaw > 1:
		return False
	else:
		#print('flaw', flaw)
		return True

#combine any sessions that got fragmented
def cleanUp(dict_):
	sortedK = sorted(dict_.keys())
	#check 3 instances at a time, prev, curr and next. 1. check timespan and change of level
	#for curr. 2. check time interval and changeof battery level between prev and curr, and 
	#between curr and next. if time diff is too small or level difference is none or continues
	#to drop then merge prev. ###First try with 2 instances instead of 3 i.e. prev and curr
	prev = dict_[sortedK[0]]
	new_ = defaultdict(list)
	c = 0
	for i in range(1, len(sortedK)):
		curr = dict_[sortedK[i]]
#		print(dict_[sortedK[i]][-1], dict_[sortedK[i]][0])
		curr_span = (dict_[sortedK[i]][-1][0] - dict_[sortedK[i]][0][0]).total_seconds()
		
		#print(curr[0][0], prev[-1][1])
		interval = (curr[0][0] - prev[-1][0]).total_seconds()
		diff = prev[-1][1] - curr[0][1]
#		if interval < 5*60*60:
#			print('prev ended at', prev[-1], diff)
#Things to check: 1. First if prev[-1] level is very close to curr[0] level or former is greater than later. 
#If true check time difference between end of prev session and start of curr session. If diff is less than
#4hrs then the two sessions should be merged.
		#if diff >= 0 or diff == -1:
			#if interval < 4*3600 or (curr_span < 5*60 and interval < 30*60):
			#	new_[sortedK[i-1]]  +=  curr
			#else:
			#	print('prev', prev[-1], '-----', 'curr', curr[0])
		#else:
		#	new_[sortedK[i]] = curr
		if prev[-1][0] > sortedK[i] or diff >= 0:
			print(i,sortedK[i-1], sortedK[i],'prev', prev[-1], '-----', 'curr', curr[0], curr[1])
			print(prev)
			print(curr)
			return
		prev = curr
		if(dict_[sortedK[i]][-1][0]-dict_[sortedK[i]][0][0]).total_seconds() < 10*60:
			print(i, sortedK[i], dict_[sortedK[i]][0][0], dict_[sortedK[i]][0][1], dict_[sortedK[i]][-1][0], dict_[sortedK[i]][-1][1])
			c += 1
	print('**', c)


#do in pool
def doInPool():
	pFile = pickle.load(open('/home/anudipa/Documents/Jouler_Extra/master_list_100.p','rb'), encoding='bytes')
	filtered = convert(pFile)
	filtered = filtered[:1]
	print(filtered)
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

#Beginning
#start_time = timeit.default_timer()
#print('*************')
all_dump = doInPool()
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

