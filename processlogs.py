#!/usr/bin/env python2
import os
import sys
import numpy as np
import matplotlib.cm as cm
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



#check folders that have no log file, get list of devices that have log
def each_device(all_, path):
    print '************', path
    localdict = Vividict()
    localdict.update(all_) 
    for root, dirs, files in os.walk(path):
	dirs.sort()
	filelist = []
	for name in files:
	    filelist.append(os.path.join(root, name))
	if len(filelist) > 0:
	    for f in filelist:
		filelist.sort()
		#print 'each files', f
		look_up_battery_tag_in_each_file(f, localdict)
		#print localdict.keys()
    all_.update(localdict)
    print '************', len(all_.keys())

#put only the direct mother directory as path, wont work otherwise
def get_device_dirs(root):
    devices = []
    for name in os.listdir(root):
	if os.path.isdir(os.path.join(root, name)):
	    #print name
	    devices.append(os.path.join(root, name))
    return devices

#things to do: go through each line, match tag line, store info as entry to dict, later pickle it
#match [dev][tag][datetime][counter, [action: ]
def look_up_battery_tag_in_each_file(dict_, path):
    #localdict = Vividict()
    #localdict.update(dict_)
    #print 'inside look_up', path
    localdict = {} 
    try:
    	_file = open(path, 'r')
    except IOError:
	print 'Cannot open ', path
	exit(0)
    for line in _file:
	data = line.split()
	if len(data) < 9 or data[8] != 'Power-Battery-PhoneLab':
	    continue
        #print line
	device = data[0]
	#localdict[device] = {}
	#treat json
	battery = ' '.join(data[9::])
        #print 'string', battery
	try:
	    _json = json.loads(battery)
	except ValueError, e:
	    print 'Not json, something wrong******************************************'
	    print path
	    print '-------------------------------------', battery
	    continue
        #print _json
	#format datetime
	t1 = data[4].split('.')[0]
        nanoSec = int(data[4].split('.')[1])
        micro = int(nanoSec/1000)
        if micro > 99999:
            curdate = data[3] + ' ' + t1 + '.' +str(micro)
        else:
            curdate = data[3]+' '+t1+'.0'+str(micro)
        t = datetime.strptime(curdate, '%Y-%m-%d %H:%M:%S.%f')
	localdict[t] = _json
    
    _file.close()
    #return localdict
    #print "#####time", len(localdict[device].keys())
    dict_.update(localdict)

def look_up_screen_tag_in_each_file(dict_, path):
    localdict = {}
    try:
	_file = open(path, 'r')
    except IOError:
	print 'Cannot open ', path
	exit(0)
    for line in _file:
	data = line.split()
	if len(data) < 9 or data[8] != 'Power-Screen-PhoneLab':
	    continue
	device = data[0]
	screen = ' '.join(data[9::])
	try:
	    _json = json.loads(screen)
	except ValueError, e:
	    print 'Not json, something wrong******************************************'
            print path
            print '-------------------------------------', screen
            continue
	t1 = data[4].split('.')[0]
        nanoSec = int(data[4].split('.')[1])
        micro = int(nanoSec/1000)
        if micro > 99999:
            curdate = data[3] + ' ' + t1 + '.' +str(micro)
        else:
            curdate = data[3]+' '+t1+'.0'+str(micro)
        t = datetime.strptime(curdate, '%Y-%m-%d %H:%M:%S.%f')
        localdict[t] = _json

    _file.close()
    #return localdict
    #print "#####time", len(localdict[device].keys())
    dict_.update(localdict)

#pickle for each device separately, spawn processes under each child process per directory
def pickle_per_dev(path):
    manager = Manager()
    _dict = manager.dict()
    dev = path.split('/')[-1]
    fileList = []
    for root, dirs, files in os.walk(path):
	dirs.sort()
        for name in files:
            fileList.append(os.path.join(root, name))
    print "****", dev, len(fileList), os.getpid()
    if len(fileList) > 0:
    	pool = Pool(processes=8)
#    	pool.map(partial(look_up_battery_tag_in_each_file, _dict), fileList)
        pool.map(partial(look_up_screen_tag_in_each_file, _dict), fileList)
	pool.close()
        pool.join()
    #print _dict
    if _dict is not None and len(_dict.keys()) > 1:
	print "!!!!!", len(_dict.keys() )
	new_ = {}
	for k in sorted(_dict.keys()):
	    new_[k] = _dict[k]
#	pFile = open('/home/anudipa/Documents/Jouler_Extra/pickles/Battery/'+dev+'.p', 'wb')
	pFile = open('/home/anudipa/Documents/Jouler_Extra/pickles/Screen/'+dev+'.p', 'wb')
    	pickle.dump(new_, pFile)
    	pFile.close()

if __name__== '__main__':
    dev = get_device_dirs('/home/anudipa/Documents/Jouler_Extra/phonelab_data_2016-03-15')
    #dev = get_device_dirs('/home/anudipa/Documents/Jouler_Extra/trial')
    print len(dev)
    if len(dev) < 1:
	sys.exit(0)
#   handling the fact that earlier run resulted in pickling of only few devices
#    root = '/home/anudipa/Documents/Jouler_Extra/pickles/Battery'
    root = '/home/anudipa/Documents/Jouler_Extra/pickles/Screen'
    fileExists = []
    for f in os.listdir(root):
	if os.path.isfile(os.path.join(root,f)):
	   d = f.split('.')[0]
	   fileExists.append(d)
    #print fileExists
    
    for d in dev:
	device = (d.split('/')[-1]).split('.')[0]
	if device in fileExists:
	   print device, "Skip"
	   continue
	print d, "started"
	pickle_per_dev(d)
	print d, "completed"
    for f in os.listdir(root):
	name = os.path.join(root, f)
	if os.path.isfile(name):
	    print name
	    p = pickle.load(open(name, 'rb'))
	    print "Loading pickle #", len(p.keys())
	    sortedK = sorted(p.keys())
	    allDays = []
	    allDays.append(sortedK[0].date())
	    for key in sortedK:
		if allDays[-1] != key.date():
		    allDays.append(key.date())
	    print len(allDays)

