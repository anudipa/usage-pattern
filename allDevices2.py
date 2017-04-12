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

#the aim of this function is to create data structure that contains discharge info
#such as current time, level, time left to charge, rate of discharge.
#from charge pickles we get list of all charging sessions
#from discharge pickle we get list of discharging timestamps with details
#then we compare both to get the closest charging event for each discharging
#timestamp and combine them to get the result dictionary
def discharge(dev):
	file1 = '/home/anudipa/Documents/Jouler_Extra/discharge2/'+dev+'.p'
	file2 = '/home/anudipa/Documents/Jouler_Extra/charge2/'+dev+'.p'
#	print(file1, file2)
	try:
#		print('Starting')
		tmp1 = pickle.load(open(file1,'rb'), encoding='bytes')
		tmp2 = pickle.load(open(file2,'rb'), encoding='bytes')
		dDataset = convert(tmp1)
		cDataset = convert(tmp2)
		ddata = dDataset[dev]
		cdata = cDataset[dev]
	except Exception as e:
		print(e)
		return
#sorted list of all days of data
	sortedD = sorted(ddata.keys())
	max_span = 0
	w1 = defaultdict(list)
	w2 = defaultdict(list)
	all_ = defaultdict(list)		#[hr] : [rate, level,time left to charge,span, datetime]
	chargeAll = []
	dict_ = {}
	start_of_charge = []
	eachCycle = {}
	track_last = ddata[sortedD[0]][0][-1]
	flag =False
#	print('Number of days:',len(sortedD))
#current time : span in seconds, time spent discharging, rate, current level, time left to charge
	for d in sortedD:
		day = d.weekday()
		listOfSessions = ddata[d]
		sortedSess = sorted(listOfSessions, key = lambda x : x[0][1])
		for i in range(len(sortedSess)):
				if i < len(sortedSess)-1 and sortedSess[i][0][1] > sortedSess[i+1][0][1]:
					print('Not sorted!!!!', sortedSess[i][0][1],  sortedSess[i+1][0][1])
		if not flag:
			track_last = sortedSess[0][-1]
			flag = True
#checking if discharging session is too many hours:
		for i in range(len(sortedSess)):
			eachSession = sortedSess[i]
			first = eachSession[0]
			ind = 1
			if eachSession[0][1] > track_last[1]:
				if (eachSession[0][1] - track_last[1]).total_seconds() < 24*60*60 :
					if eachSession[0][0] <= track_last[0] :
						first = track_last
						ind = 0
				else:
#					checkCycle(eachCycle)
					eachCycle = {}
			track_last = eachSession[-1]
			for k in range(len(eachSession)-1):
				if eachSession[k][1] > eachSession[k+1][1]:
					print('Not Sorted!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
					break
			for j in range(ind,len(eachSession)):
#				if eachSession[j][1] > datetime.datetime(2015,9,12,20,0,0) and eachSession[j][1] < datetime.datetime(2015,9,13,2,0,0):
#					if j < len(eachSession) -1 :
#						print(first, eachSession[j], eachSession[j+1])
#					else:
#						print(first, eachSession[j])
				drop = first[0] - eachSession[j][0]
				if drop >= 0:
					diff = ((eachSession[j][1] - first[1]).total_seconds()/60.0)
					#find number of hours
					total_hours = int(diff/60)
					if drop == 0 or total_hours == 0:
						rate = 0
						per_hour_drop = 0
					else:
						rate = drop/diff
						per_hour_drop = drop/total_hours
						
					level_for_this_hour = eachSession[j][0]
					#current_time = eachSession[j][1]
					t = 0
					while(t <= total_hours):
						current_time = eachSession[j][1] - timedelta(hours= t)			#recreate from last to first
						w1[current_time] = [rate,int(level_for_this_hour), current_time ]
						eachCycle[current_time] = w1[current_time]
						level_for_this_hour = level_for_this_hour + per_hour_drop
						t += 1
					if drop > 0:
						first = eachSession[j]				
				else:
					w1[eachSession[j][1]] = [0.0, eachSession[j][0], eachSession[j][1]]
					if drop < 0:
						first = eachSession[j]
						
					eachCycle[eachSession[j][1]] =  w1[eachSession[j][1]]
#get charging instances from discharging sessions:
#	sortedK = sorted(dict_.keys())
	sortedK = sorted(w1.keys())
	chargeAll = []
	start = False
	for i in range(len(sortedK)-1):
		if not start and w1[sortedK[i+1]][1] > w1[sortedK[i]][1]:
			if len(chargeAll) == 0 or ((sortedK[i] - chargeAll[-1][2]).total_seconds() > 10*60 and w1[sortedK[i]][1] <= chargeAll[-1][3]):
				chargeAll.append([sortedK[i], w1[sortedK[i]][1], sortedK[i+1], w1[sortedK[i+1]][1]])
#				chargeAll.append([sortedK[i], w1[sortedK[i]][1], -1, -1])
			elif w1[sortedK[i+1]][1] >= chargeAll[-1][3]:
				chargeAll[-1][2] = sortedK[i+1]
				chargeAll[-1][3] = w1[sortedK[i+1]][1]
			start = True
		elif start:
			if w1[sortedK[i]][1] >= w1[sortedK[i-1]][1] and w1[sortedK[i]][1] >= chargeAll[-1][3]:
				chargeAll[-1][2] = sortedK[i]
				chargeAll[-1][3] = w1[sortedK[i]][1]
			elif w1[sortedK[i]][1] > w1[sortedK[i+1]][1]:
				start = False
				if chargeAll[-1][1] > chargeAll[-1][3]:
					print('FLAGGED!', chargeAll[-1])
	
#	print('***************************************************************')
#	print(len(chargeAll))
	count = 0
	last = 0
	discharging = []
	for i in range(0,len(chargeAll)):
		start_ = chargeAll[i][0]
		start_level = chargeAll[i][1]
		end_ = chargeAll[i][2]
		end_level = chargeAll[i][3]
#		print('###',start_level, end_level)
		wrong = False
		last_bin = -1 
		rate_ = defaultdict(list)
		tmp_list = []
		startB = False
		start_d = sortedK[last]
		for j in range(last, len(sortedK)):
#check if j_time is less than start of charge and battery level at j_time is less than or equal to
#that at j-1_time or else there are two possibilities, either it is harmless fluctuation or an
#unregistered charging event took place. if former case, ignore the j_time reading  or else ignore
#the whole discharging session. Todo: update start time of missing charge event as j_time and calculate
#time left to charge accordingly
			if j < len(sortedK)-1 and sortedK[j+1] <= start_:
				if (w1[sortedK[j]][1] < w1[sortedK[j+1]][1] and (w1[sortedK[j+1]][1]-w1[sortedK[j]][1]) >= 10) or (sortedK[j+1] - sortedK[j]).total_seconds() >= 24*60*60:
					rate_ = defaultdict(list)
					tmp_list = []
					last = j
					start_d = sortedK[last]
					startB = False
					continue
			if sortedK[j] <= start_ :
				if not startB:
					startB = True
				start_d = sortedK[j]
				if len(tmp_list) > 0:
					if (sortedK[j] - tmp_list[-1]).total_seconds() > 2*60*60 :
						print('!!!!!',sortedK[j], tmp_list[-1], start_)
						print('*****', w1[sortedK[j]][1], w1[tmp_list[-1]][1], start_level, dev)
						fillintheblanks(tmp_list[-1], w1[tmp_list[-1]][1], sortedK[j], w1[sortedK[j]][1], w1, tmp_list)
						
				tmp_list.append(sortedK[j])
				time_left_to_charge_mins = int((start_ - sortedK[j]).total_seconds()/60.0)
				bins = int(time_left_to_charge_mins / 60)
				rate_[bins].append(w1[sortedK[j]])
#				if not last_bin == bins and abs(bins - last_bin) > 9:
#					if not last_bin == -1:
#						print(last_bin, bins, sortedK[j], w1[sortedK[j]][1], sortedK[j-1], w1[sortedK[j-1]][1], start_)
#				last_bin = bins
				#calculate time spent discharging. total_span = time_discharging + time_left_charging
				time_discharging_mins = int((sortedK[j] - sortedK[last]).total_seconds()/60.0)
				total_span = time_discharging_mins + time_left_to_charge_mins
				rate_[bins][-1].append(time_discharging_mins)			#index: 3
				rate_[bins][-1].append(total_span)				#index: 4
				#add the battery levels at start and end of charge event
				rate_[bins][-1].append(start_level)				#index: 5
				rate_[bins][-1].append(end_level)				#index: 6
				#to identify records from the same session			 index: 7
				rate_[bins][-1].append(start_)
			elif sortedK[j] > start_ :
				if startB:
					if len(tmp_list) > 0 and not start_d == tmp_list[-1]:
						print('Not consistent')
					if (w1[tmp_list[-1]][1] - start_level) > 1:
						print('!!!!!',tmp_list[-1], w1[tmp_list[-1]][1], start_, start_level)
					mid = ((tmp_list[-1] - tmp_list[0]).total_seconds()/60)/2.0
					pivot = tmp_list[0]+timedelta(minutes=mid)
					mid_ = min(tmp_list, key=lambda d: abs(d - pivot))
#					if abs(mid_ - pivot).total_seconds()/60 > 60:
#						print(pivot, mid_, w1[mid_][1], sortedK[last], w1[sortedK[last]][1], start_d, w1[start_d][1])
#						print(tmp_list)
					mid_level = w1[mid_][1]
					start_level = w1[tmp_list[0]][1]
					end_level = w1[tmp_list[-1]][1]

					if (start_level - end_level) == 0 or (start_level - mid_level) == 0:
						rate_firstH_min = 0.0
						rate_secondH_min = 0.0
					elif mid_ ==  start_d or mid_ == sortedK[last]:
						rate_firstH_min = rate_secondH_min = (start_level - end_level)/abs((sortedK[last] - start_d).total_seconds()/60.0)
						#print('is it too small?', (start_d - sortedK[last]).total_seconds()/60, start_level, end_level, len(tmp_list))
					else:
						rate_firstH_min = (start_level - mid_level)/((mid_ - sortedK[last]).total_seconds()/60.0)
						rate_secondH_min = (mid_level - end_level)/((start_d - mid_).total_seconds()/60.0)
					discharging.append([tmp_list[0], start_level, rate_firstH_min, mid_, mid_level, rate_secondH_min, tmp_list[-1], end_level, total_span])
					tmp_list = []
					startB = False
				if sortedK[j] >= end_:
					last = j
					break
		#now add the temp entries to the mothership
		if len(rate_.keys()) > 0:
			#print(sorted(rate_.keys()))
			#checking consistency of total_span. Remove after success
			#total_span should be same all across. +-2 mins is expected since there is approximation
			expected = rate_[0][0][4]
			for bins in rate_.keys():
				list_ = rate_[bins]
				for e in list_:
					all_[bins].append(e)
					if abs(e[4] - expected) > 3:
						print('inconsistency!!!!', expected, e, bins)
			
	dict_ = {'binned': all_, 'summary': discharging}
	return dict_

def checkCycle(each_):
	sortedK = sorted(each_.keys())
	if len(sortedK) == 0:
		print('Nothing to check!')
		return
#	print(sortedK)
	for i in range(len(sortedK)-1):
		if abs(sortedK[i]- sortedK[i+1]).total_seconds() > 60*60 :
			print('*****', sortedK[i], each_[sortedK[i]], sortedK[i+1], each_[sortedK[i+1]])
#			print('#####', sorted(each_))
#			break


def fillintheblanks(start_time, start_level, end_time, end_level, dict_, list_):
	drop = start_level - end_level
	diff = (end_time - start_time).total_seconds()/60.0
	if drop < 0:
		print('Check later')
		return
	#find number of hours
	if int(diff/60) == 0 or drop == 0:
		total_hours =0
		per_hour_drop = 0
		rate = 0
	else:
		total_hours = int(diff/60)
		per_hour_drop = drop/total_hours
		rate = drop/diff
	level_for_this_hour = end_level
	#current_time = eachSession[j][1]
	t = 0
	while(t <= total_hours):
		current_time = end_time - timedelta(hours= t)                  #recreate from last to first
		dict_[current_time] = [rate,int(level_for_this_hour), current_time ]
		list_.append(current_time)
		print('filling in', current_time, int(level_for_this_hour), t, per_hour_drop, drop)
		level_for_this_hour = level_for_this_hour + per_hour_drop
		t += 1

#function to execute discharge() computation for all devices in parallel that have seen more than 100 days of data
def doInPool():
	#masterlist is a list of devices that have more than 100 days of data.
	pFile = pickle.load(open('/home/anudipa/Documents/Jouler_Extra/master_list_100.p','rb'), encoding='bytes')
	#this is only because the preproccessed pickles are done with python 2 which is incompatible with python3
	filtered = convert(pFile)
	filtered = filtered[:20]
	print(filtered)
	pool = Pool(processes=4)
	res = pool.map(discharge, filtered)
	pool.close()
	pool.join()
	return res

##start of the active code
#to track computation time
start_time = timeit.default_timer()
print('*************')
all_dump = doInPool()
#print(len(all_dump))
#all_ = discharge('cdfce167c0f7fbcdc14b841f0cf4cd6d7fe6d470')
#all_ = discharge('dcac21d5d407e60ee3fc3999982d30399bcd1f91')
elapsed = timeit.default_timer() - start_time
print('Elapsed time:',elapsed)

#sys.exit()

count_low_level_trigger = 0
dataX = []
dataY = []
for i in range(len(all_dump)):
	all_ = all_dump[i]['binned']
#	print('For device #',i)
#1. get the mean battery level seen at start of charging session
	temp = []
	for j in range(len(all_[0])):
		temp.append(all_[0][j][5])
	common_level = np.mean(sorted(temp))
#2. get the number of sessions > 180 mins and the number of sessions where it reached low_level early
	trackAll_ = []
	track_ = []
	max_t = 0
	sortedK = sorted(all_.keys(), reverse=True)
	for t_left in sortedK:
		list_ = all_[t_left]
		each = []
		t_left_mins = t_left*60
		for j in range(len(list_)):
#			if list_[j][4] > list_[j][1]:
#				print(list_[j])
#				continue
			span_in_mins = list_[j][4]
			level_tgt = abs(list_[j][1] - list_[j][5])
			if span_in_mins > 60*3 :
				if len(trackAll_) == 0 or list_[j][7] not in trackAll_:
					trackAll_.append(list_[j][7])
				if  t_left_mins > 0.5*span_in_mins and level_tgt <= 5:
					each.append(level_tgt)
					if t_left_mins > max_t:
						max_t = t_left_mins
					if len(track_) == 0 or list_[j][7] not in track_:
						track_.append(list_[j][7])
	perc = (len(track_)/len(trackAll_))*100
#	print('For device#',i,'---mean starting level: ',common_level,'---percentage of early drain: ', perc)
	#dataX.append(common_level)
	#dataY(perc)
#3.
for i in range(len(all_dump)):
	all_ = all_dump[i]['summary']
	count1 = 0
	count2 = 0
	count3 = 0
	count4 = 0
	dataX = []
	dataY = []
	ratio = []
#[sortedK[last], start_level, rate_firstH_min, mid_, mid_level, rate_secondH_min, sortedK[j], end_level, total_span]
	for j in range(len(all_)):
		t_drop = all_[j][1] - all_[j][7]
		h_drop = all_[j][1] - all_[j][4]
		t_time = (all_[j][6] - all_[j][0]).total_seconds()/60.0
		h_time = (all_[j][3] - all_[j][0]).total_seconds()/60.0
		if h_drop == t_drop or t_drop == 0:
			continue
		if all_[j][8] > 3*60:  
			count1 += 1
#			print(j, h_drop/t_drop, h_drop, t_drop)
			if (h_drop/t_drop) >= 0.7:
				#print(all_[j][2], all_[j][5], all_[j][1], all_[j][4], all_[j][7], t_drop, h_drop)
				if all_[j][2] > 10 or all_[j][5] > 10:
					print(all_[j])
				ratio.append(all_[j][2]/all_[j][5])
				count2 += 1
			elif (h_drop/t_drop) >= 0.4:
				count3 += 1
			elif h_drop/t_drop <= 0.25:
				count4 += 1
	perc = (count2/count1)*100
	print('Percentage:', perc, count1, count2, count3, count4)
#ax.set_xticklabels(xlabels)
#ax.set_xlabel('Time left to charge (hours)')
#ax.set_ylabel('Levels left to reach charge event')
#fig.show()
elapsed = timeit.default_timer() - start_time
print('Elapsed Time:', elapsed)
