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
from collections import OrderedDict
from functools import partial
import pickle
from multiprocessing import Process, Manager, current_process, Pool
import pprint

#to created nexted dict on the fly. refer stackoverflow
class Vividict(dict):
    def __missing__ (self, key):
        value = self[key] = type(self)()
        return value

#load pickles for number of devices mentioned, if 0 then for all
#_all = defaultdict(list)
def loadAllCharge(root, num): 
    devices = []
    for f in os.listdir(root):
        name = os.path.join(root, f)
        if os.path.isfile(name):
	    devices.append(name)
#    for name in os.walk(root):
#        devices.append(os.path.join(root, name))
    if num > 0 and num < len(devices):
	del devices[num::]
    else:
	print "Total: ", len(devices)
    all_ = {}
    c = 1
#    for d in devices:
#	device = (d.split('/')[-1]).split('.')[0]
#	print c, device
#	temp = loadOne(d)
#	if temp == -1:
#	    continue
#	all_[device] = eachChargeSpan(temp)
#	c+=1
    if len(devices) > 0:
	pool = Pool(processes=8)
	res = pool.map(loadOne, devices)
	pool.close()
	pool.join()
	print 'results', len(res)
   	#pFile = open('/home/anudipa/Documents/Jouler_Extra/All_Charge_Data.p','wb')
    	#pickle.dump(res, pFile)
    	#pFile.close()
        for i in range(len(res)):
	    if type(res[i]).__name__ != 'dict':
		print i, res[i]
		continue
            for dev in res[i]:
                if (res[i][dev].keys()) > 1:
                    pFile = open('/home/anudipa/Documents/Jouler_Extra/charge/'+dev+'.p','wb')
                    pickle.dump(res[i], pFile)
                    pFile.close()


def loadOne(path):
    try:
        dict_ = pickle.load(open(path, 'rb'))
    except:
        print path
        return -1
    device = (path.split('/')[-1]).split('.')[0]
    all_ = {}
    if len(dict_.keys()) > 100*100:
        all_[device] = getChargeSessions(dict_, [0,1,2,3,4,5,6])
    else:
        all_[device] = {}
    print device, "Days-->", len(all_[device].keys())
    return all_


#dictionary a[time]=[start_time, end_time, start_level, end_level]
def getChargeSessions(dict_, days):
    #all_ = defaultdict(list)
    all_ = {}
    sortedK = sorted(dict_.keys())
    isCharging = False
    start = -1
    end = -1
    level = -1
    allDays = []
    lastDate = sortedK[0].date()
    allDays.append(lastDate)
    #print "first date", lastDate
    for timestamp in sortedK:
	#print timestamp.date().weekday(), timestamp.date()
	#if timestamp.date().weekday() not in days and isCharging is False:
	#    print timestamp.date(), timestamp.date().weekday()
	#    continue
	if 'BatteryProperties' not in dict_[timestamp].keys():
	    continue
	level = dict_[timestamp]['BatteryProperties']['Level']
	currentTime = timestamp.time()
	currentDate = timestamp.date()
        if currentDate not in all_.keys():
            all_[currentDate] = []

	if currentDate != lastDate:
	    if currentDate < lastDate:
		print 'Error!!!!!!!!!'
	    #print "new date", currentDate
	    if isCharging:
		currentDate = lastDate
	    else:
	    	lastDate = currentDate
	    if currentDate not in allDays:
		allDays.append(currentDate)
	#print dict_[timestamp]['BatteryProperties']['Status']
	try:
	    if isCharging == False:
		if dict_[timestamp]['BatteryProperties']['Status'] == 'Charging' or dict_[timestamp]['BatteryProperties']['Status'] == 2:
		    isCharging = True
#		    start = currentTime
		    start = timestamp
		    all_[currentDate].append([start,level,end, -1])
	    else:
#		end = currentTime
		end = timestamp
  	    	all_[currentDate][-1][2] = end
		all_[currentDate][-1][3] = level
		if dict_[timestamp]['BatteryProperties']['Status'] == 'Discharging' or dict_[timestamp]['BatteryProperties']['Status'] > 2:
                    isCharging = False 
		    start = -1
		    end = -1

	except ValueError, IOError:
	    print "Error", timestamp, dict_[timestamp]
    #print '****', all_.keys()
    #print 'Days --> ', len(allDays)
    cleaned_all = cleanUp(all_)
    if len(cleaned_all.keys()) > 200:
        return cleaned_all
    else:
        return {}

#if two corresponding sessions are less than 5 mins apart then merge into a single one,
#this is to remove jitters, maintain consistency and remove false positives for charging sessions
def cleanUp(dict_):
    if (len(dict_.keys())) < 1:
	return {} 
    new_ = {}
    
    for dates in dict_:
	i = 0
	j = 1
	if dates not in new_.keys():
	    new_[dates] = []
	while i < (len(dict_[dates])-2):
	    t1 = dict_[dates][i][2]
	    j = i+1
	   # new_[dates].append(dict_[dates][i])
	    entry = dict_[dates][i]
	    while j < (len(dict_[dates])-1):
	    	t2 = dict_[dates][j][0]
#	    	delta = (t2.hour*3600+t2.minute*60+t2.second) - (t1.hour*3600+t1.minute*60+t1.second)
		delta = (t2 - t1).seconds
	    	#print t1, t2, delta
	    	if delta < 60*5:
		    entry[2] = dict_[dates][j][2]
		    entry[3] = dict_[dates][j][3]
		    t1 = entry[2] 
		    j = j+1
	    	else:
		    break
	    new_[dates].append(entry)
	    i = j
	if i < len(dict_[dates]):
	    new_[dates].append(dict_[dates][-1])
	   
    return new_

#get features like average charging span, distribution of charging events over time
def eachChargeSpan (dict_):
    eachSpan = {}
    if dict_ is None:
	return None
    for i in range(7):
	eachSpan[i] = []
    sortedK = sorted(dict_.keys())
    for key in sortedK:
	day = key.weekday()
	if len(dict_[key]) < 1:
	   continue
	for val in dict_[key]:
	    if val[2] == -1 or val[3] == -1:
		continue
#	    span = (val[2].hour*3600+val[2].minute*60+val[2].second) - (val[0].hour*3600+val[0].minute*60+val[0].second)
	    span = (val[2] - val[0]).seconds
	    if span < 0:
		print val
	    eachSpan[day].append([span, val[1], val[3], val[0].hour, val[2].hour])
    return eachSpan

def plotChargeSpan(dict_):
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.set_xlim([0,24])
    ax1.set_ylim([0,1200])
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.set_xlim([0,24])
    ax2.set_ylim([0,1000])
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)
    ax3.set_xlim([0,24])
    colors = iter(cm.rainbow(np.linspace(0, 1, len(dict_.keys()))))
    for device in dict_:
	x1 = []
	x2 = []
	y1 = []
	y2 = []
	cdfWeekday = []
	for day in range(7):
	    for val in dict_[device][day]:
		if day < 5:
		    x1.append(val[3])
		    y1.append(val[0]/60)
		else:
		    x2.append(val[3])
		    y2.append(val[0]/60)
	c = next(colors)
	if len(x1) > 30:
	    xvals = np.sort(x1)
	    yvals  = np.arange(len(xvals))/float(len(xvals))
	    ax1.scatter(x1,y1,color=c, alpha=0.5)
	    ax3.plot(xvals, yvals, color=c)
	if len(x2) > 30:
	    ax2.scatter(x2,y2,color=c, alpha = 0.5)

    fig1.savefig("Weekday.pdf", dpi=300)
    fig2.savefig("Weekend.pdf", dpi=300)
    fig3.savefig("CDF_Weekday.pdf", dpi=300)

#for screen
def loadAllScreen(root, num):
    devices = []
    for f in os.listdir(root):
        name = os.path.join(root, f)
        if os.path.isfile(name):
            devices.append(name)
#    for name in os.walk(root):
#        devices.append(os.path.join(root, name))
    if num > 0 and num < len(devices):
        del devices[num::]
    else:
        print "Total: ", len(devices)
    all_ = {}
    c = 1
#    for d in devices:
#        device = (d.split('/')[-1]).split('.')[0]
#        print c, device
#        temp = loadOneScreen(d)
#        if temp is None:
#            continue
#        all_[device] = temp
#        c+=1
    if len(devices) > 0:
	pool = Pool(processes=8)
	res = pool.map(loadOneScreen, devices)
	pool.close()
	pool.join()
	print 'result', len(res) 
    	pFile = open('/home/anudipa/Documents/Jouler_Extra/All_Screen_Data.p','wb')
    	pickle.dump(res, pFile)
    	pFile.close()

def loadOneScreen(path):
    try:
        dict_ = pickle.load(open(path, 'rb'))
    except:
        print path
        return -1
    device = (path.split('/')[-1]).split('.')[0]
    all_ = {}
    all_[device] = getSessions(dict_)
    print device, len(all_[device].keys())
    return all_

def getSessions(dict_):
    all_ = defaultdict(list)
    allDays = []
    sortedK = sorted(dict_.keys())
    inSession = False
    lastDate = sortedK[0].date()
    allDays.append(lastDate)
    for timestamp in sortedK:
	if "Action" not in dict_[timestamp].keys():
	    continue
	
        currentTime = timestamp.time()
        currentDate = timestamp.date()
        if currentDate != lastDate:
            if currentDate < lastDate:
                print 'Error!!!!!!!!!'
            #print "new date", currentDate
            if inSession:
                currentDate = lastDate
            else:
                lastDate = currentDate
            if currentDate not in allDays:
                allDays.append(currentDate)
	try:
	    if inSession == False:
		if dict_[timestamp]['Action'] == 'android.intent.action.SCREEN_ON':
		    inSession = True
		    all_[currentDate].append([timestamp, -1])
	    else:
		if dict_[timestamp]['Action'] == 'android.intent.action.SCREEN_OFF':
		    inSession = False
		    all_[currentDate][-1][1]= timestamp
        except ValueError, IOError:
            print "Error", timestamp, dict_[timestamp]
    #print '****', all_.keys()
    print 'Days --> ', len(allDays)
    return all_

#**********************************************************************************
#**********************************************************************************
#*********************************EXTRACT FEATURES*********************************
#**********************************************************************************
#**********************************************************************************


