#!/usr/bin/env python2
import os
import sys
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from sklearn import svm

#to-do: do svm classifier for a. all discharge points, b. all charge points and c. discharge + charge
#opt: 1, 2, 3

def predictForDevice(device, day, opt):
#load pickle for given device
    if opt < 1 or opt > 3:
	print 'Wrong Options!!'
	return
    all_data = loadP(device, day, opt)
    all__ = process_data(all_data)
    training_data = all__['train']
    test_data = all__['test']


def loadP(device, day, opt):
    _all = {}
    if opt == 1:
	filename = '/home/anudipa/Documents/Jouler_Extra/discharge/'+device+'.p'
    elif opt == 2:
	filename = '/home/anudipa/Documents/Jouler_Extra/charge/'+device+'.p'
    try:
	dict_ = pickle.load(open(filename, 'rb'))
    except :
	print 'Filename not found'
	return None
    for dates in dict_[device].keys():
	if dates.weekday() != day:
	    continue
	_all[dates] = dict_[device][dates]
    return _all

def process_data(all_data):
    _all = {}
    train_processed = [[] for i in range(24)]
    test_data = []
    total_days = len(all_data.keys())
    num_test = total_days/10
    num_train = total_days - num_test
    sortedDates = sorted(all_dates.keys())
    for d in range(0,num_train-1) :
	list_of_sessions = all_data[sortedDates[d]]
	for i in range(len(list_of_sessions)):
	    each_session = list_of_sessions[i]
	    for k in range(len(each_session)):
		hr = each_session[k][1].hour
		level = each_session[k][0]
		train_processed[hr].append(level)
    for d in range(num_train, total_days):
	temp = [[] for t in range(24)]
	for each_session in all_data[sortedDates[d]:
	    for i in range(len(each_session)):
		hr = each_session[i][1].hour
		l = each_session[i][0]
		temp[hr].append(l)
	for i in range(24):
	    if len(temp[i]) < 1:
	  	temp[hr].append(100)
	test_data.append(temp)
   
	test_data.append(
    _all['train'] = train_processed
    _all['test'] = test_data
    return _all	



