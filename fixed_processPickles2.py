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
import pickle
from multiprocessing import Process, Manager, current_process, Pool
from matplotlib.backends.backend_pdf import PdfPages


def loadAllDischarge(root, num):
    devices = []
    for f in os.listdir(root):
        name = os.path.join(root, f)
        if os.path.isfile(name):
            devices.append(name)
    if num > 0 and num < len(devices):
        del devices[num::]
        print(devices)
    else:
        print("Total: ", len(devices))
    if len(devices) > 0:
        pool = Pool(processes=8)
        res = pool.map(loadOneDischarge, devices)
        pool.close()
        pool.join()
        print('results', len(res))
    for i in range(len(res)):
        for dev in res[i]:
            if len(res[i][dev].keys()) > 1:
                pFile = open('/home/anudipa/Documents/Jouler_Extra/discharge/'+dev+'.p','wb')
                pickle.dump(res[i], pFile)
                pFile.close()
        #pFile = open('/home/anudipa/Documents/Jouler_Extra/200_Discharge_Data.p','wb')
        #pickle.dump(res, pFile)
        #pFile.close()


def loadOneDischarge(path):
    try:
        device = (path.split('/')[-1]).split('.')[0]
        dict_ = pickle.load(open(path, 'rb'))
    except:
        print(path)
        return {device:{}} 
    #device = (path.split('/')[-1]).split('.')[0]
    all_ = {}
    if len(dict_.keys()) > 100*100:
        all_[device] = getDischargeSessions(dict_, [0,1,2,3,4,5,6])
    else:
        all_[device] = {}
        print(device, "Days-->", len(all_[device].keys()))
    return all_

def getDischargeSessions(dict_, days):
#    all_ = defaultdict(list)
    all_ = {}
    sortedK = sorted(dict_.keys())
    isDischarging = False
    lastLevel = -1
    level = -1
    for i in range(len(sortedK)):
        timestamp = sortedK[i]
        if 'BatteryProperties' not in dict_[timestamp].keys():
            continue
        level = dict_[timestamp]['BatteryProperties']['Level']
        currentTime = timestamp.time()
        currentDate = timestamp.date()
        if currentDate not in all_.keys():
            all_[currentDate] = []
        try:
            if isDischarging == False:
                if dict_[timestamp]['BatteryProperties']['Status'] == 'Discharging' or dict_[timestamp]['BatteryProperties']['Status'] > 2:
                    isDischarging = True
                    all_[currentDate].append([[level, timestamp]])
                    lastLevel = level
                    #all_[currentDate].append([start,level,end, -1])
            else:
#               end = currentTime
                #end = timestamp
                #all_[currentDate][-1][2] = end
                #all_[currentDate][-1][3] = level
                if len(all_[currentDate]) < 1:
                    all_[currentDate].append([[level, timestamp]])
                elif level <= lastLevel:
                    all_[currentDate][-1].append([level,timestamp])
                    lastLevel = level
                if dict_[timestamp]['BatteryProperties']['Status'] == 'Charging' or dict_[timestamp]['BatteryProperties']['Status'] == 2:
                    isDischarging = False
                    lastLevel = -1

        except (ValueError, IOError):
            print("Error", timestamp, dict_[timestamp])
    #print '****', all_.keys()
    #print 'Days --> ', len(allDays)
    cleaned_all = cleanUpDischarge(all_)
    if len(cleaned_all.keys()) > 200:
        return cleaned_all
    else:
        return {}

def cleanUpDischarge(dict_):
    if (len(dict_.keys())) < 1:
        return {}
    #new_ = defaultdict(list)
    new_ = {}
    for dates in dict_:
        if len(dict_[dates]) < 1:
            continue
        if dates not in new_.keys():
            new_[dates] = []
        currentSession = dict_[dates][0]
        for i in range(1,len(dict_[dates])-1):
            if len(dict_[dates][i]) < 2:
                continue
            end = currentSession[-1][1]
            start = dict_[dates][i][0][1]
            delta = (start - end).seconds
            if delta < 60*2 and currentSession[-1][0] <= dict_[dates][i][0][0]:
                currentSession+=dict_[dates][i]
            else:
                new_[dates].append(currentSession)
                currentSession = dict_[dates][i]

        new_[dates].append(currentSession)

    return new_


def plotDischargeAll():
    filtered = []
    path = '/home/anudipa/Documents/Jouler_Extra/discharge'
    for root, dirs, files in os.walk(path):
        filelist = []
        for name in files:
            filelist.append(os.path.join(root, name))
        if len(filelist) > 0:
            for f in filelist:
                all_ = pickle.load(open(f, 'rb'))
                dev = all_.keys()[0]
                #print '#', dev, len(all_[dev].keys())  
                if len(all_[dev].keys()) > 330:
                    filtered.append(all_)
    print("Number of devices filtered", len(filtered))
    #pp1 = PdfPages("/home/anudipa/Documents/Jouler_Extra/Discharge")
    total = len(filtered)
    for day in range(7):
#   fig1 = plt.figure(0,dpi=100)
#   ax1 = fig1.add_subplot(111)
#   ax1.set_title("Start Discharge Sessions For Day #"+day)
#   ax1.set_xlim([0,24])

#        fig2 = plt.figure(0,dpi=100)
#        ax2 = fig2.add_subplot(111)
#        ax2.set_title("End Discharge Sessions For Day #"+day)
#        ax2.set_xlim([0,24])

#   fig3 = plt.figure(0,dpi=100)
#        ax3 = fig3.add_subplot(111)
#        ax3.set_title("Span Discharge Sessions For Day %s" % day)
#        ax3.set_xlim([0,24])
#   ax3.set_ylim([0,24])

#   fig4 = plt.figure(1,dpi=100)
#        ax4 = fig4.add_subplot(111)
#        ax4.set_title("Rate Discharge Sessions For Day %s" % day)
#        ax4.set_xlim([0,24])


        colors = iter(cm.rainbow(np.linspace(0, 1, total)))
        for i in range(len(filtered)):
            dev = filtered[i].keys()[0]
            rate = defaultdict(list)
            span = defaultdict(list)
            startH = []
            endH = []

            dates_to_check = []
            for dates in filtered[i][dev].keys():
                if dates.weekday() == day:
                    dates_to_check.append(dates)
            sortedDates = sorted(dates_to_check)
            print(dev, '#', day, len(sortedDates))
            for j in range(len(sortedDates)):
                now = sortedDates[j]
                list_of_sessions = filtered[i][dev][now]
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
                                        print(last_hour, ':', level[r], level[r+1], '--', t[r], t[r+1])
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

            #for x in range(len(rate)):
            #print x, 'hr:', max(rate[x])
            c = next(colors)
            #print 'Rates', dev
            for hr in rate.keys():
                x = [hr for e in range(len(rate[hr]))]
                y = rate[hr]
                #print hr, y 
                #ax4.scatter(x, y, color=c, alpha=0.4)
                avg = np.mean(rate[hr])
                ax4.scatter(hr, avg, color=c, marker=(5, 1))
            ax4.set_ylabel('Rate per minute')
            ax4.set_xlabel('Hours')

            #print 'Span', dev
            for hr in span.keys():
                x = [hr for e in range(len(span[hr]))]
                y = span[hr]
                #print hr, y
                #ax3.scatter(x, y, color=c, alpha=0.4)
                avg = np.mean(span[hr])
                ax3.scatter(hr, avg, color=c, marker=(5, 1))
            ax3.set_ylabel('Discharge Session Length in Hour')
            ax3.set_xlabel('Hours')
            break
        #pp1.savefig(fig3)
        #pp1.savefig(fig4)
        #plt.close()
        fig3.savefig('Spanday %s .pdf' % day, dpi=100)
        fig4.savefig('Rateday %s .pdf' % day, dpi=100)
        plt.close('all')

    #pp1.close()
    all_ = []

def plotDischargeEach():
    filtered = []
    path = '/home/anudipa/Documents/Jouler_Extra/discharge'
    for root, dirs, files in os.walk(path):
        filelist = []
        for name in files:
            filelist.append(os.path.join(root, name))
        if len(filelist) > 0:
            for f in filelist:
                all_ = pickle.load(open(f, 'rb'))
                dev = all_.keys()[0]
                #print '#', dev, len(all_[dev].keys())  
                if len(all_[dev].keys()) > 330:
                    filtered.append(all_)
    print("Number of devices filtered", len(filtered))
    total = len(filtered)
#pdf pages for each device
    pp1 = PdfPages("/home/anudipa/Documents/Jouler_Extra/SpanHist")
    pp1 = PdfPages("/home/anudipa/Documents/Jouler_Extra/ChargeLeft")
