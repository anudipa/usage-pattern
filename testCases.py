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

def testDischarging(dev, source):
	data_type = type(source)
	if data_type not dict or data_type not defaultdict:
		print('Error! Not Compatible Datatype! Use disctionary')
		return
	if len(dev) < 8:
		print('Not valid device, Atleast enter first 8 alphanumeric string')
		return
	
