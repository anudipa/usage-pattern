#!/usr/bin/env python2
import os
import sys
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from pylab import plot, show
from datetime import *
from collections import defaultdict
import pickle
from multiprocessing import Process, Manager, current_process, Pool
from matplotlib.backends.backend_pdf import PdfPages
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std

#load pickle into dictionary

def loadP(path):
    try:
	all_ = pickle.load(open(path, 'rb'))
	print len(all_)
    except:
	print 'Error', path
	return None
    return all_

def doInPool():
    chargeS = loadP('/home/anudipa/Documents/Jouler_Extra/All_Charge_Data.p')
    if chargeS is None:
        print 'Empty'
        return None
    filtered = []
    for i in range(len(chargeS)):
	try:
	    temp = chargeS[i]
	    device = (temp.keys())[0]
	    print device, len(temp[device].keys())
	    if len(temp[device].keys()) > 180:
	    	filtered.append(temp)
	except:
	    print 'Error in accessing device'
    print 'No. of devices in filtered list --> ', len(filtered)
    if len(filtered) > 0:
	pool = Pool(processes=8)
	res = pool.map(get_discharge_cycles, filtered)
	pool.close()
	pool.join()
	print 'Final Results', len(res)

#Step 1: Get list of device names that have around one year worth of data. add them
#	 to a filter master list
#Step 2: For each device get charge cycles and discharge cycles info in two dictionaries
#Step 3: Which hours in the day for each weekday has high probability to see long charge
#	 event and discharge event and the how likely a long charge event is followed by
#	 long discharge event
#Step 4: for each day when there is likely to be spike in discharge rates, are they long
#	 term or short spikes
#Step 5: aim is to classify as many main usage pattern as possible. then use it as a
#	 checklist to predict 

def get_filtered_masterlist():
#Either use charge data or discharge data
    filtered = []
    path_charge = '/home/anudipa/Documents/Jouler_Extra/charge'
    #path_discharge = ''
    for root, dirs, files in os.walk(path_charge):
	filelist = []
	for name in files:
	    filelist.append(os.path.join(root, name))
	if len(filelist) > 0:
	    for f in filelist:
		all_ = pickle.load(open(f, 'rb'))
		dev = all_.keys()[0]
		if len(all_[dev].keys()) > 100:
		    filtered.append(dev)
    print "Number of devices filtered: ", len(filtered)
    pFile = open('/home/anudipa/Documents/Jouler_Extra/master_list_100.p', 'wb')
    pickle.dump(filtered, pFile)
    pFile.close()
    #print filtered


    return filtered
#for only discharge
def perDeviceDetails():
    filtered = get_filtered_masterlist()
    path_charge = '/home/anudipa/Documents/Jouler_Extra/charge/'
    path_discharge = '/home/anudipa/Documents/Jouler_Extra/discharge/'
    pdf1 = PdfPages('/home/anudipa/Documents/Jouler_Extra/deviceDischarge')
    for each in filtered:
	filename = path_discharge+each+'.p'
	try:
	    dict_ = pickle.load(open(filename, 'rb'))
	except:
	    print 'File not found', each
	    continue
	dev = dict_.keys()[0]
	rate = defaultdict(list)
	span = defaultdict(list)
	startH = []
	endH = []
	for day in range(7):
	    dates_to_check = []
	    for dates in dict_[dev].keys():
		if dates.weekday() == day:
		    dates_to_check.append(dates)
	    if len(dates_to_check) < 20:
		continue
	    sortedDates = sorted(dates_to_check)
	    print dev, '#', day, len(sortedDates)
	    fig1, ax = plt.subplots(4, sharex=True)
	    ax[0].set_title("For Device %s on Day # %s" % (dev, str(day)))
	    for j in range(len(sortedDates)):
                now = sortedDates[j]
                list_of_sessions = dict_[dev][now]
                for k in range(len(list_of_sessions)):
                    each_session = list_of_sessions[k]
                    if len(each_session) < 2:
                        #print dev, sortedDates[j], 'Ignoring'
                        continue
                    startH.append(each_session[0][1].hour)
                    endH.append(each_session[-1][1].hour)
                    delta = ((each_session[-1][1] - each_session[0][1]).seconds)/(3600.0)
                    span[each_session[0][1].hour].append(delta)
                    #get discharge rate for each hour
                    last_hour = each_session[0][1].hour
                    level = []
                    t = []
                    for x in range(len(each_session)):
                        if last_hour == each_session[x][1].hour:
                            level.append(each_session[x][0])
                            t.append(each_session[x][1])
                        else:
                            if len(level) < 2:
                                #print "could not compute rate", dev, sortedDates[j]
                                rate_ = 0.0
                            else:
                                #print "successful", dev, sortedDates[j]
                                #calculate rate as level drop per second
                                for r in range(len(level)-1):
                                    drop = level[r] - level[r+1]
                                    diff = (t[r+1] - t[r]).seconds
                                    #if drop > 10:
                                        #print last_hour, level
                                        #print last_hour, t
                                    if drop > 0 and diff > 0:
                                        rate_ = (drop/float(diff))
                                    else:
                                        rate_ = 0.0
                                    if rate_ == 1.0:
                                        print last_hour, ':', level[r], level[r+1], '--', t[r], t[r+1]
                                    rate[last_hour].append(rate_)
                            last_hour = each_session[x][1].hour
                            level = []
                            t = []
                    if len(level) > 1:
                        for r in range(len(level)-1):
                            drop = level[r] - level[r+1]
                            diff = (t[r+1] - t[r]).seconds
                            if drop > 0 and diff > 0:
                                rate_ = (drop/float(diff))
                            else:
                                rate_ = 0.0
                            rate[last_hour].append(rate_)
	    ax[0].hist(startH, 24, normed='1', facecolor='red')
	    #ax[0].set_xlabel('Hours in the day')
	    ax[0].set_ylabel('Start of D.S.')

	    ax[1].hist(endH, 24, normed='1', facecolor='green')
	    ax[1].set_ylabel('End of D.S.')

	    for hr in rate.keys():
		x = [hr for e in range(len(rate[hr]))]
	 	y = rate[hr]
		avg = np.mean(rate[hr])
		ax[2].scatter(x, y, color='blue', marker=(5,1))
		ax[2].plot(hr, avg, 'yo')
	    ax[2].set_ylabel('level drop/sec')
	    ax[2].set_xlim([0,24])

	    for hr in span.keys():
                x = [hr for e in range(len(span[hr]))]
                y = span[hr]
                avg = np.mean(span[hr])
                ax[3].scatter(x, y, color='magenta', marker=(5,1))
                ax[3].plot(hr, avg, 'yo')
            ax[3].set_ylabel('Session Length')
	    ax[3].set_xlim([0,24])
	    ax[3].set_xlabel('Hours')

	    pdf1.savefig(fig1)
	    plt.close('all')
    pdf1.close()

#for only charge
def perDeviceCharge():
    filtered = get_filtered_masterlist()
    path_charge = '/home/anudipa/Documents/Jouler_Extra/charge/'
    pdf1 = PdfPages('/home/anudipa/Documents/Jouler_Extra/deviceCharge')
    for each in filtered:
        filename = path_charge+each+'.p'
        try:
            dict_ = pickle.load(open(filename, 'rb'))
        except:
            print 'File not found', each
            continue
        dev = dict_.keys()[0]
        span = defaultdict(list)
        startH = []
        endH = []
        for day in range(7):
            dates_to_check = []
            for dates in dict_[dev].keys():
                dates_to_check.append(dates)
            if len(dates_to_check) < 20:
                continue
            sortedDates = sorted(dates_to_check)
            print dev, '#', day, len(sortedDates)
            fig1, ax = plt.subplots(4, sharex=True)
            ax[0].set_title("For Device %s on Day # %s" % (dev, str(day)))
#            for i in range(len(sortedDates)):


#get battery pattern for any day[0-6], any device:
def batterypatten(dev, day):
    if day < 0 or day > 6:
	print 'Wrong day!!!'
	return
    f_discharge = '/home/anudipa/Documents/Jouler_Extra/discharge/'+dev+'.p'
    f_charge = '/home/anudipa/Documents/Jouler_Extra/charge/'+dev+'.p'
    try:
	_dictD = pickle.load(open(f_discharge,'rb'))
	_dictC = pickle.load(open(f_charge, 'rb'))
    except:
	print("Unexpected error:", sys.exc_info()[0])
    	raise
#2D array each element is an array of battery levels seen in that hour== idx
    _allHours = [[] for i in range(24)]
    dates_to_check = []
    for d in _dictD.keys():
	for dates in _dictD[d].keys():
	    if dates.weekday() == day:
		dates_to_check.append(dates)
	    if len(dates_to_check) == 30:
		break
    sortedDates = sorted(dates_to_check)
    print sortedDates
    print dev, '#', day, '-->', len(sortedDates), ':', len(_dictD[d].keys())
    fig, ax = plt.subplots(figsize=(8,6))
    color=iter(cm.rainbow(np.linspace(0,1,len(sortedDates))))
    for i in range(len(sortedDates)):
	eachDay = []
	temp = [[] for t in range(24)]
        now = sortedDates[i]
	print '#', i, ':', now
	list_of_sessions = _dictD[d][now]
	for j in range(len(list_of_sessions)):
	    each_ = list_of_sessions[j]
	    for k in range(len(each_)):
	        hr = each_[k][1].hour
		level = each_[k][0]
		_allHours[hr].append(level)
		temp[hr].append(level)
        #print temp
	for k in range(24):
	    if len(temp[k]) > 0:
	    	x = np.mean(temp[k])
		eachDay.append(x)
	    else:
		eachDay.append(100)
	print eachDay
	c=next(color)
	ax.plot(eachDay, color=c, ls='--')
    #ax.set_xlim([0,24])
    #ax.set_ylim([0,101])
		
#calculate probability of a level at each hour
    
    #_avgDay = []
    
	#print _allHours[hr]
	#sorted_ = sorted(_allHours[hr], reverse=True)
	#prob_ = np.arange(len(sorted_))/(len(sorted_) -1)
	#print hr, sorted_  
	#for i in range(len(prob_)):
	#    if prob_[i] >= 0.75:
	#        print hr, sorted_[i]
	#	_avgDay.append(sorted_[i])
	#	break
	#hist, edges = np.histogram(_allHours[hr], bins=np.arange(0,101,10))
	#print hist, edges
	#idx = np.where(hist==max(hist))
	#print max(hist)
	#for x in idx:
	#    common = edges[x]
	#    _avgDay.append(common)
	#    print hr, max(common)
	#_avg = np.mean(_allHours[hr])
	#_avgDay.append(_avg)
	#print hr, _avg
	    


#clustering
#def lookfordamnpatterns():
#    list_of_dev = get_filtered_masterlist()
#    filtered = list_of_dev[::20]
#    path_discharge = '/home/anudipa/Documents/Jouler_Extra/charge/'
#    for each in filtered:
#        filename = path_discharge+each+'.p'
#        try:
#            dict_ = pickle.load(open(filename, 'rb'))
#        except:
#            print 'File not found', each
#            continue
#        dev = dict_.keys()[0]
	#End level: X; End Hour: Y; Highest Rate: Z
#	X = []
#	Y = []
#	Z = []
