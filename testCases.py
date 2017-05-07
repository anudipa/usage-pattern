#!/usr/bin/env python

#this library is to include all test cases 
#to confirm consistency of all data concerned
#with user behavior

import os
import numpy as np
from multiprocessing import Process, current_process, Pool
from collections import defaultdict, Counter
import pickle
import json
from datetime import *
from scipy.stats import itemfreq
import timeit


#Testcase 1: To test if in a given dictionary for all discharging sessions
# the battery levels and timestamp are consistent
# Error: Increasing battery level
# Error: Decreasing timestamp, end of previous session is after start of next session
# Error: Time difference between two consecutive discharging session is shorter than 10 mins or does not effect any increase in battery level
# Error: Ending battery level of prev session is less than starting battery level of the next session

def testDischarging(dev, source, opt=False):
	data_type = type(source)
#	if data_type != dict or data_type != collections.defaultdict:
#		print('Error! Not Compatible Datatype! Use dictionary not ', data_type)
#		return False
	if len(dev) < 8:
		print('Not valid device, Atleast enter first 8 alphanumeric string')
		return
	sortedK = sorted(source.keys())
	if len(sortedK) < 10:
		print('Possible Error: too small of a dataset')

	countE = 0
	prev = sorted(source[sortedK[0]], key=lambda x: x[0])
	for i in range(1, len(sortedK)):
		curr = sorted(source[sortedK[i]], key=lambda x:x[0])
		curr_span = (curr[-1][0] - curr[0][0]).total_seconds()
		charge_span = (curr[0][0] - prev[-1][0]).total_seconds()
		charge_diff = (curr[0][1] - prev[-1][1])
		#if charging span or discharging span is less than 10 mins or level diff due to charge is less than 3 it can be a potential error
		if curr_span <= 10*60 or charge_span <=10*60 or charge_diff < 3:
			#print('**Error: too short : ', i, curr_span/60, charge_span/60, charge_diff, prev[-1], curr[0], curr[-1])
			#print('**Error: ', sortedK[i-1], sortedK[i], curr[-1])
			if opt:
				return False
			countE += 1
		if curr[0][0] < prev[-1][0]:
			#print('**Error: inconsistency in consecutive session: ', sortedK[i-1], prev[-1], sortedK[i], curr[0])
			if opt:
				return False
			countE += 1
		#check each span
		for j in range(len(curr)-1):
			if curr[j][1] < curr[j+1][1]:
				#print('**Error: discharging data not consistent: ', curr[j], curr[j+1])
				if opt:
					return False
				countE += 1
		prev = curr
	if not opt:
		print('*Total error count: ', countE)
	print('*End*')
	return False		#change this back to True //MUST//


