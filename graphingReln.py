#!/usr/bin/env python

import os
import numpy as np
from multiprocessing import Process, current_process, Pool
from collections import defaultdict, Counter
import pickle
from datetime import *
from pylab import *
from scipy.stats import itemfreq
import testCases
import dischargingRate as dR
import statistics as stats


#the generic pool function
def doInPool():
	all_dump = dR.doInPool()
	for d in range(len(all_dump)):
		dev = next(iter(all_dump[d]))
		dict_ = all_dump[d][dev]
		print(dev, 'no of sessions: ', len(dict_.keys()))
		startbatteryData(dict_, dev)

#for different battery level start, how does other features differ
def startbatteryData(dict_, dev):
	#get start levels for each session, span, total drop in charge, variance in discharging rate
	data1 = defaultdict(list)
	data2 = defaultdict(list)
	sortedK = sorted(dict_.keys())
	for i in range(len(sortedK)):
		start_level 	= dict_[sortedK[i]][0][1]
#		total_drop	= (dict_[sortedK[i]][0][1] - dict_[sortedK[i]][-1][1])
		drop		= dict_[sortedK[i]][-1][1]
		span_in_min	= (dict_[sortedK[i]][-1][0] - dict_[sortedK[i]][0][0]).total_seconds()/60.0
		rates_		= [dict_[sortedK[i]][j][2] for j in range(1, len(dict_[sortedK[i]]))]
		var_rate	= stats.pvariance(rates_)
		data1[start_level].append([drop, span_in_min, var_rate])
		data2[int(start_level/10)].append([drop, span_in_min, var_rate])
	#check data
	#bin 10
	for bins in data2.keys():
		drop = [data2[bins][i][0] for i in range(len(data2[bins]))]
		span = [data2[bins][i][1] for i in range(len(data2[bins]))]
		var_rate = [data2[bins][i][2] for i in range(len(data2[bins]))]
		print(drop)
		print('Bin:', bins, '-----------> var drop:', stats.pvariance(drop), '; var span:', stats.pvariance(span), '; mean variance:', stats.mean(var_rate))
