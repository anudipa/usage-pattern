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
from hmmlearn import hmm


def convert(data):
	data_type = type(data)
	if data_type == bytes : return data.decode()
	if data_type in (str,int,float): return data
	if data_type in (datetime.datetime,datetime.date): return data
	if data_type == dict: data = data.items()
	return data_type(map(convert, data))


def getObservationData(dev):
	file1 = '/home/anudipa/Documents/Jouler_Extra/discharge2/'+dev+'.p'
	file2 = '/home/anudipa/Documents/Jouler_Extra/charge2/'+dev+'.p'
#       print(file1, file2)
	try:
#               print('Starting')
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
	all_ = defaultdict(list)                #[hr] : [rate, level,time left to charge,span, datetime]
	chargeAll = []
	start_of_charge = []
	for d in sortedD:
		day = d.weekday()
		listOfSessions = ddata[d]
		track_first = listOfSessions[0][0]
		track_last = listOfSessions[0][-1]
#checking if discharging session is too many hours:
		for i in range(len(listOfSessions)):
			eachSession = listOfSessions[i]
			if i > 0:
				if (eachSession[0][1] - track_last[1]).total_seconds() < 30*60 and  eachSession[0][0]<=track_last[0] :
					track_last = eachSession[-1]
				else:
                                        track_first = eachSession[0]
                                        track_last = eachSession[1]
                        eachCycle = []
                        levels = []
                        timeD = []
                        span = (track_last[1] - track_first[1]).total_seconds()
                        if span/60 > max_span:
                                max_span = span/60
                        if track_first != eachSession[0]:
                                first = listOfSessions[i-1][-1]
                                ind = 0
                        else:
                                first = eachSession[0]
                                ind = 1
                        for j in range(ind,len(eachSession)):
                                drop = first[0] - eachSession[j][0]
                                if drop > 0:
                                        diff = ((eachSession[j][1] - first[1]).total_seconds()/60.0)
                                        rate = drop/diff
                                        #first = eachSession[j]
                                        #dict_[eachSession[j][1]] = [span, (eachSession[j][1] - eachSession[0][1]).total_seconds()/60.0, rate, eachSession[j][0],0,0]
##TODO: temporarily for each time slot alot rate of discharge seen. temporary assumption
                                        #find number of hours
                                        if first[1].hour == eachSession[j][1].hour:
                                                total_hours =0
                                                per_hour_drop = 0
                                        else:
                                                total_hours = abs(eachSession[j][1].hour - first[1].hour)
                                                per_hour_drop = int(drop/total_hours)
                                        level_for_this_hour = eachSession[j][0]
                                        #current_time = eachSession[j][1]
                                        t = 0
                                        while(t <= total_hours):
                                                current_time = eachSession[j][1] - timedelta(hours= t)                  #recreate from last to first
                                                w1[current_time] = [rate,level_for_this_hour, (current_time - track_first[1]).total_seconds()]
                                                level_for_this_hour = level_for_this_hour + per_hour_drop
                                                t += 1
                                        first = eachSession[j]

#get charging instances from discharging sessions:
#       sortedK = sorted(dict_.keys())
        sortedK = sorted(w1.keys())
        chargeAll = []
        start = False
        for i in range(len(sortedK)-1):
                if not start and w1[sortedK[i+1]][1] > w1[sortedK[i]][1]:
                        if len(chargeAll) == 0 or ((sortedK[i] - chargeAll[-1][2]).total_seconds() > 10*60 and w1[sortedK[i]][1] <= chargeAll[-1][3]):
                                chargeAll.append([sortedK[i], w1[sortedK[i]][1], sortedK[i+1], w1[sortedK[i+1]][1]])
#                               chargeAll.append([sortedK[i], w1[sortedK[i]][1], -1, -1])
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

#       print('***************************************************************')
#       print(len(chargeAll))
        count = 0
        last = 0

        for i in range(0,len(chargeAll)):
                start_ = chargeAll[i][0]
                start_level = chargeAll[i][1]
                end_ = chargeAll[i][2]
                end_level = chargeAll[i][3]
#               print('###',start_level, end_level)
                wrong = False
                last_bin = -1
                rate_ = defaultdict(list)
                for j in range(last, len(sortedK)):
                        if j < len(sortedK)-1 and sortedK[j+1] <= start_:
                                if (w1[sortedK[j]][1] < w1[sortedK[j+1]][1] and (w1[sortedK[j+1]][1]-w1[sortedK[j]][1]) >= 10) or (sortedK[j+1] - sortedK[j]).total_seconds() >= 24*60*60:
                                        rate_ = defaultdict(list)
                                        last = j
                                        continue
                        if sortedK[j] <= start_ :
#                               if (start_ - sortedK[j]).total_seconds()/3600 > 30:
#                                       print('Anomaly: ', sortedK[j], w1[sortedK[j]][1], '-------------', start_, start_level, end_)

                                #print(j,sortedK[j],'-->charged between: ', start_, end_)
                                time_left_to_charge_mins = int((start_ - sortedK[j]).total_seconds()/60.0)
                                bins = int(time_left_to_charge_mins / 60)
                                rate_[bins].append(w1[sortedK[j]])
#                               if not last_bin == bins and abs(bins - last_bin) > 9:
#                                       if not last_bin == -1:
#                                               print(last_bin, bins, sortedK[j], w1[sortedK[j]][1], sortedK[j-1], w1[sortedK[j-1]][1], start_)
#                               last_bin = bins
                                #calculate time spent discharging. total_span = time_discharging + time_left_charging
                                time_discharging_mins = int((sortedK[j] - sortedK[last]).total_seconds()/60.0)
                                total_span = time_discharging_mins + time_left_to_charge_mins
                                rate_[bins][-1].append(time_discharging_mins)                   #index: 3
                                rate_[bins][-1].append(total_span)                              #index: 4
                                #add the battery levels at start and end of charge event
                                rate_[bins][-1].append(start_level)                             #index: 5
                                rate_[bins][-1].append(end_level)                               #index: 6
                                #to identify records from the same session                       index: 7
                                rate_[bins][-1].append(start_)
                        elif sortedK[j] >= end_:
                                last = j
#                               if wrong:
#                                       count += 1
#                                       print(start_, end_, start_of_charge[i])
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
        return all_






def hmmModel(obs):

	
