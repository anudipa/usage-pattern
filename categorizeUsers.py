#!/usr/bin/env python2
import os
import sys
import numpy as np
from pylab import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from multiprocessing import Process, current_process, Pool
import pickle
import json
from datetime import *
from itertools import cycle
from collections import Counter, defaultdict
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import MeanShift, estimate_bandwidth
#from hmmlearn import hmm

#calculate 4 dimensions for each users ( batterylevel group every 10, time every 4 hrs):
#w - probability/variance of batterylevel group repeating a time group [when dev is discharging]
#x - probability/variance of charging sessions repeating a time group
#y - probability/variance of total time device spends discharging in a day
#z - probability/variance of same batterylevel group at the start of charging sessions
#a - average number of charging sessions per day
#b - average charging session length per day
# 0-9 10-19 20-29 30-39 40-49 50-59 60-69 70-79 80-89 90-99,100 [add the data on the fence to both groups, fence is +-3]
# 0-3 4-7 8-11 12-15 16-19 20-23 [ add data on fence to both groups, fence is +-15 mins]

def loadOne(device):
    #dataPts = defaultdict(lambda: defaultdict(list))
    dataPts = {}
    try:
	path1 = '/home/anudipa/Documents/Jouler_Extra/discharge/'+device+'.p'
	path2 = '/home/anudipa/Documents/Jouler_Extra/charge/'+device+'.p'
	ddict_ = pickle.load(open(path1,'rb'))
	cdict_ = pickle.load(open(path2, 'rb'))
	dev = ddict_.keys()[0]
	ddata_ = ddict_[dev]
	cdata_ = cdict_[dev]
	dataPts[dev] = computeProbability(ddata_,cdata_)
 	#print dev, '*****************************************'
#	for attr in dataPts[dev].keys():
#	    print attr, dataPts[dev][attr]
    except Exception as e:
	#print "WTF ",sys.exc_info()[0] 
	print 'WTF', e, sys.exc_info()[0]
	return None
    return dataPts


def loadAll(num):
    print '#',num
#    devices = []
    res = []
    root = '/home/anudipa/Documents/Jouler_Extra/discharge/'
    master = pickle.load(open('/home/anudipa/Documents/Jouler_Extra/master_list_100.p','rb'))
    devices = []
    for i in range(num):
	if i > len(master)-1:
	    break
	devices.append(master[i])
#    for f in os.listdir(root):
#	name = os.path.join(root, f)
#	if os.path.isfile(name):
#	    devices.append(f)
#	if len(devices) == num:
#	    break
    print "All devices about to be processed *********"
    print len(devices), '****', devices
    if len(devices) > 0:
    	pool = Pool(processes=8)
	res = pool.map(loadOne, devices)
	pool.close()
	pool.join()
    
    return res

#todo: compute highest frequency for each date and combine to compute the final overall
def computeProbability(discharge_, charge_):
    if len(discharge_) < 1 or len(charge_) < 1:
	return None
#    print 'Starting'
#w,x,y, z for each of 7 days
    w = [0 for i in range(7)]
    x = [0 for i in range(7)]
    y = [0 for i in range(7)]
    z = [0 for i in range(7)]
    a = [0 for i in range(7)]
    b = [0 for i in range(7)]
    days = {}
    spans = [[] for i in range(7)]
#    print 'Starting'
    for each_date in discharge_.keys():
	temp = []
	listOfSessions = discharge_[each_date]
	day = each_date.weekday()
	if days is None or day not in days.keys():
	    days[day] = {}
	for i in range(len(listOfSessions)):
	    each_session = listOfSessions[i]
	    if len(each_session) < 2:
		continue
	    delta = ((each_session[-1][1] - each_session[0][1]).seconds)/(60.0)
	    temp.append(delta)
	    for j in range(len(each_session)):
		batteryG = (each_session[j][0])/10
		remainder = (each_session[j][0])%10
		timeG = (each_session[j][1].hour)/4
		if timeG not in days[day].keys():
		    days[day][timeG] = []
		days[day][timeG].append(batteryG)
#accounting for border cases. eg. 88 or 91 shd be in both groups 80-89 and 90-99
		if remainder != 0:
		    if remainder < 3 and batteryG > 0:
			days[day][timeG].append(batteryG-1)
		    elif remainder > 7 and batteryG < 9:
			days[day][timeG].append(batteryG+1)
		
	spans[day].append(sum(temp))
    for day in days.keys():
#	highest = -1
# two ways to calculate: 
#1. get the highest perc when a battery level group frequently appeared 
#   for a particular time interval  
#2. get the average perc of times most frequent battery level for 
#   different time intervals
#Alt. calculate variance of battery levels for each time group and then
#	calculate percentage of time low variance is seen
	varW = []
 	for timeG in days[day].keys():
	    #common = Counter(days[day][timeG]).most_common()[0][0]
#	    percW = ((Counter(days[day][timeG]).most_common()[0][1])/float(len(days[day][timeG])))*100.0
	    #percW.append(((Counter(days[day][timeG]).most_common()[0][1])/float(len(days[day][timeG])))*100.0)
	    varW.append(np.var(days[day][timeG]))
#	    print day, timeG, len(days[day][timeG]), perc, highest
#	    if percW > highest:
#		highest = percW
	    #print day, timeG, perc
#	w[day] = highest
	percW = 0
	for i in varW:
	    if i < 11:
		percW +=1
	w[day] = 100.0*(percW/float(len(varW)))
	#percY = 100.0*((Counter(spans[day]).most_common()[0][1])/float(len(spans[day])))
	varY = np.var(spans[day])
	y[day] = varY			#percY
#    print 'Done with discharging session'
#charging sessions *********************
    days = {}
    for d in range(7):
	days[d] = [[],[],[],[]]
    for each_date in charge_.keys():
    	listOfSessions = charge_[each_date]
	if len(listOfSessions) == 0:
	    continue
	day = each_date.weekday()
#trimming listofsessions, many 100 to 100 charging session is repeated
	trimmed = 0	#length of trimmed list
	trimmedList = []
	trimmedList.append(listOfSessions[0])
	lastT = listOfSessions[0][2]
	lastL = listOfSessions[0][3]
	for i in range(1,len(listOfSessions)):
	    each_ = listOfSessions[i]
	    if lastL < each_[1] and lastT < each_[0]:
		trimmedList[-1][2] = each_[2]
		trimmedList[-1][3] = each_[3]
	    else:
		trimmedList.append(each_)
	    lastT = each_[2]
	    lastL = each_[3]
	for i in range(len(trimmedList)):
	    each_session = trimmedList[i]
	    if (each_session[1] == 100 and each_session[3] == 100) or (each_session[3] < each_session[1]):
		continue
	    trimmed += 1
	    timeG = (each_session[0].hour)/4
	    remainder = (each_session[0].hour)%4
	    days[day][0].append(timeG)
	    if each_session[2] != -1:
	        session_len = (each_session[2] - each_session[0]).seconds/60.0
	        days[day][3].append(session_len)
		
	    if remainder == 3:
	        if timeG < 5:
		    days[day][0].append(timeG+1)
		else:
		    days[(day+1)%7][0].append(0)
	    days[day][1].append(each_session[1])
	days[day][2].append(trimmed)
        if trimmed > 15:
            #print listOfSessions
            print len(listOfSessions), len(trimmedList), trimmed, '************************************'

#    print days.keys()
    for day in days.keys():
#        x[day] = 100.0*((Counter(days[day][0]).most_common()[0][1])/float(len(days[day][0])))
	x[day] = np.var(days[day][0])
#        z[day] = 100.0*((Counter(days[day][1]).most_common()[0][1])/float(len(days[day][1])))
	z[day] = np.var(days[day][1])
	#if z[day] > 100:
	    #print Counter(days[day][1]).most_common()[0], len(days[day][1])
        a[day] = np.mean(days[day][2])
        b[day] = np.mean(days[day][3])
#	print day, x[day], z[day], a[day], b[day]


    all_points = defaultdict(list)
    all_points['w'] = w
    all_points['x'] = x    
    all_points['y'] = y
    all_points['z'] = z
    all_points['a'] = a
    all_points['b'] = b
    #print a
#    print 'Ending'

    return all_points


def checkUp(num):
    all_ = loadAll(num)
    #lets see whats cooking
    for i in range(len(all_)):
	if all_[i] is None:
	    continue
	dev = all_[i].keys()[0]
	print dev, '***************************->'
	data = all_[i][dev]
	if data is None:
	    continue


#def classifySVC(data):
    


def clustering(opt):
    if opt < 0 or opt > 4:
	print 'get a grip!! choose between 0,1,2,3,4'
	return 
    num = 1000
    all_ = loadAll(num)
    devices = [all_[i].keys()[0] for i in range(len(all_))]
    names = ['group1', 'group2', 'group3']
    
    #for any day preprocessing into dataset [a, b, w, x, y, z]
    if opt == 0:
    	for day in range(7):
	    data = []
    	    for i in range(len(all_)):
	    	dev = all_[i].keys()[0]
	    	temp = all_[i][dev]
	    	data.append([temp['a'][day], temp['b'][day], temp['w'][day], temp['x'][day], temp['y'][day], temp['z'][day]])
	    #data.append([temp['w'][day], temp['z'][day], temp['a'][day]]) 
	kmeansGraph3d(data, day)
	return
    elif opt == 1:
	data = {}
	for day in range(7):
	    data[day] = []
	    for i in range(len(all_)):
		dev = all_[i].keys()[0]
		temp = all_[i][dev]
		data[day].append([temp['a'][day], temp['b'][day], temp['w'][day], temp['x'][day], temp['y'][day], temp['z'][day]])
	kmeansGraph2d(data, devices)
    elif opt == 2:
	data = []
	dev_list = []
	for day in range(7):
	    #if day == 0 or day == 6:
		#continue
	    
	    #data = []
	    for i in range(len(all_)):
		dev = all_[i].keys()[0]
                temp = all_[i][dev]
		if temp['a'][day] > 50:
		    print '!!!!!!!!!', dev, day, temp['a'][day], '!!!!!!!!!!!!!'
                data.append([temp['a'][day], temp['b'][day], temp['w'][day], temp['x'][day], temp['y'][day], temp['z'][day]])
		#data.append([temp['a'][day], temp['x'][day], temp['z'][day]])
		dev_list.append(dev)    
	meanshiftGraph(np.array(data), devices, 2)   #last argument for day if doing the other way as in per weekday type 0= weekdays: 1=weekends: 2=alldays 3=one day 

def kmeansGraph3d(dataset, day):
    km = KMeans(n_clusters=3, init='k-means++', n_init=10, max_iter=300, tol=0.0001, random_state=0)
    reduced_data = PCA(n_components=3).fit_transform(dataset)
#    y_km = km.fit_predict(dataset)
#    y_pred = km.fit_predict(dataset)
    y_pred = km.fit_predict(reduced_data)
    y = np.choose(y_pred, [0, 1, 2]).astype(np.float)
    print day, y
    l = len(reduced_data)
    X = np.reshape(reduced_data,(l,3)) 
    fig = figure(day, dpi=300)
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
#    ax = fig.add_subplot(111,projection='3d')
    print y_pred
    print y
    ax.scatter(X[:,0], X[:,1], X[:,2], c=y)
    ax.set_xlabel('Feature 1', fontsize=4)
    ax.set_ylabel('Feature 2', fontsize=4)
    ax.set_zlabel('Feature 3', fontsize=4)
    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    #ylabel('ChargeSession rpt. over T')
    #tick_params(axis="both", labelsize='small')
    title('KMeans for Day # %d' % (day))
    #fig.set_size_inches(4.0, 3.5)
    figname = ('kmeans3d%d.pdf' % (day))
    #fig.savefig(figname, dpi=300)
    return

def kmeansGraph2d(dataset, devices):
    km = KMeans(n_clusters=3, init='k-means++', n_init=10, max_iter=300, tol=0.0001, random_state=0)
    fig = figure(1, dpi=300)
    ax = Axes3D(fig)
    markers = ['o','^','*','h','8','p','d']
    colors = ['blue', 'red', 'green']
    scores = [[0,0,0] for i in range(len(devices))]
    for day in range(7):
	reduced_data = PCA(n_components=2).fit_transform(dataset[day])
	y_pred = km.fit_predict(reduced_data)
	keep_scores(list(y_pred), scores)
 	temp =  np.full(len(reduced_data), day, dtype=np.int)
	add_on = temp.reshape(len(reduced_data),1)
	#print y_pred
	y = np.choose(y_pred, [0, 1, 2]).astype(np.float)
	modified_X = np.hstack((reduced_data, add_on))
	#ax.scatter(modified_X[:,0], modified_X[:,2], modified_X[:,1], c=y, marker=markers[day])
	bar0 = list(y_pred).count(0)
        bar1 = list(y_pred).count(1)
	bar2 = list(y_pred).count(2)
	#print bar0, bar1, bar2
	ax.bar([0,10,20], [bar0,bar1,bar2], day, zdir='y', color=colors, alpha=0.5)
    #print scores
    bars_score = get_highest_scorer(scores,devices)
    print bars_score, sum(bars_score)
    ax.bar([3,13,23,27],bars_score, 7, zdir='y', color = ['blue', 'red', 'green', 'yellow'], alpha=0.8)
    ax.set_xlabel('Clusters', fontsize=6)
    ax.set_ylabel('Days', fontsize=6)
    ax.set_zlabel('Number of devices', fontsize=6)
    #ax.w_xaxis.set_ticklabels([])
    #ax.w_zaxis.set_ticklabels([])
    fig.savefig('kmeans2dAllDays_1.pdf', dpi=300)
    return


def meanshiftGraph(X, devices, day):
    dev_label = []
    #n = 75*len(X)/100
    
    bandwidth = estimate_bandwidth(X, quantile=0.2)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(X)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    print("number of estimated clusters : %d" % n_clusters_)
    print labels
    fig = figure(day,dpi=100)

    colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    for k, col in zip(range(n_clusters_), colors):
    	my_members = labels == k
    	cluster_center = cluster_centers[k]
    	plot(X[my_members, 0], X[my_members, 1], col + '.')
    	plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=14)
    title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()
    
    #which device is in which cluster
    n = len(X)
    print 'X', n, len(X[0])
    d = 0
    if day == 0:
	d = 5
    elif day == 1:
	d = 2
    else:
	d = 7
    groups = [[] for i in range(n_clusters_)]
    j = 0
    for i in range(0,n,d):
	scores = Counter(labels[i:i+d])
	cl = scores.most_common(1)[0]
	o = scores.most_common(1)[0][1]
	c = scores.most_common(1)[0][0]
	groups[c].append(devices[j])
	dev_label.append(cl)
	j += 1
    #print len(dev_label), dev_label
    print len(labels)
#    for i in range(len(devices)):
#	print devices[i], dev_label[i]

    print groups
    for i in range(n_clusters_):
	print len(groups[i])


def keep_scores(y, scores):
    for i in range(len(y)):
	cluster = y[i]
	if cluster > 2 or cluster < 0:
	    print 'WTF: Something is wrong!!'
	scores[i][cluster] += 1
    return

def get_highest_scorer(scores,devices):
    bars = [0,0,0,0]
    dev = [[],[],[],[]]
    for i in range(len(scores)):
	highest = max(scores[i])
	if highest < 4:
	    bars[-1] += 1
	    dev[3].append(devices[i])
	else:
	    ind = scores[i].index(highest)
	    bars[ind] += 1
	    dev[ind].append(devices[i])
    print dev
    return bars

#****************************************************************************************************************
#************************************CLASSIFICATION**************************************************************
#****************************************************************************************************************
#classify days
def classifyEachUser2(dev):
    file_ = '/home/anudipa/Documents/Jouler_Extra/discharge/'+dev+'.p'
    print file_
    try:
	dataset = pickle.load(open(file_,'rb'))
	ddata = dataset[dev]
    except Exception as e:
	print 'Error', e
	return
    total = len(ddata.keys())
    train_num = (80*total)/100
    print total, train_num
    #X_train = [[] for i in range(7)]
    #y_train = [[] for i in range(7)]
    X_train = []
    y_train = []
    #X_test = [[] for i in range(7)]
    #y_test = [[] for i in range(7)]
    X_test = []
    y_test = []
    y = [0,1,2,3,4,5,6]
    count = 0
    sortedD = sorted(ddata.keys())
    trainingD = sortedD[:train_num]
    testingD = sortedD[train_num:]
    for d in range(len(trainingD)):
	day = trainingD[d].weekday()
	    
	listOfSessions = ddata[trainingD[d]]
	for i in range(len(listOfSessions)):
	    eachSession = listOfSessions[i]
	    pre_rate = -1.0
	    post_rate = -1.0
	    for j in range(len(eachSession)):
		hr = eachSession[j][1].hour
		level = eachSession[j][0]
#get discharge get pre and post the level
		if j == 0:
		    if i > 1:
			pre_rate = post_rate
		else:
		    diff = abs(eachSession[j-1][0] - eachSession[j][0])
		    drop = (eachSession[j][1] - eachSession[j-1][1]).seconds
		#X_train[day].append(level)			#battery level for per hour: assumption
		#y_train[day].append(hr)
		X_train.append([level, hr])
		y_train.append(day)

    for d in range(len(testingD)):
        day = testingD[d].weekday()

        listOfSessions = ddata[testingD[d]]
        for i in range(len(listOfSessions)):
            eachSession = listOfSessions[i]
            for j in range(len(eachSession)):
                hr = eachSession[j][1].hour
                level = eachSession[j][0]
                X_test.append([level,hr])                      #battery level for per hour: assumption
                y_test.append(day)

    #poly_svc = SVC(kernel='poly', degree=3)
    #rbf_svc = SVC(kernel='rbf', gamma=0.5)
    for day in range(7):
	fig, ax = plt.subplots(2,sharex = True, sharey = True)
	ax[0].scatter(y_train[day], X_train[day],c=y_train)
	ax[0].set_xlimit([0,24])
	ax[0].set_ylimit([0,101])
	ax[0].set_ylabel('Battery Level')
	ax[0].set_xlabel('Hour')
	poly_svc = SVC(kernel='poly', degree=3).fit(X_train, y_train)
        for dates in testingD:
	     Z = poly_svc.predict(X_test[day])
	     ax[1].plot(Z, X_test[day],'k')
	     ax[1].plot(y_test[day], X_test[day], 'r--')
	ax[1].set_xlimit([0,24])
	ax[1].set_ylimit([0,101])
        ax[1].set_ylabel('Battery Level')
    return

def classifyAll(list_):
    if len(list_) == 0:
	return
    pool = Pool(processes=8)
    res = pool.map(classifyForEach, list_)
    pool.close()
    pool.join()

    fig = figure(1,dpi=100)
    discarded = 0
    for i in range(len(res)):
	#print '#',i
	scores = res[i]
#	if np.mean(scores) < 0.2:
#	    discarded += 1
#	    continue
	plot(range(1,6),scores, 'r-')
#    print 'Discarded', discarded

def classifyForEach(dev):
    allData_ = createDataset(dev)
    time_test= []
    X_train = []
    y_train = []
    X_test = []
    y_test = []

    scores = []
    #print 'Device:', dev

    for i in range(24):
	for j in range(4):
	    time_test.append(i*60+j*15)
#dividing into training and testing --> take 75% of total data as training and rest as test
    #clf = svm.SVC()
#lets do for only weekdays
    for i in range(1,6):
	total = len(allData_[i])
    	nTrain = (75*total)/100
    	nTest = total - nTrain
	X = []
	y = []
	for j in range(total):
	    time_combine = allData_[i][j][1]*60+allData_[i][j][2]*15
	    X.append(time_combine)
            y.append([allData_[i][j][0],allData_[i][j][3], allData_[i][j][4]])

	    if j == nTrain:
		X_train.append(np.array(X))
		y_train.append(np.array(y))
		#print len(X), len(y)
		#print X, y
		X = []
		y = []
	    elif j == total-1:
		#print len(X), len(y)
		X_test.append(np.array(X))
                y_test.append(np.array(y))
	break
    for i in range(5):
        #print 'Day',i
	#print X_train[i].shape, y_train[i].shape
	#print len(y_train[i]), len(X_train[i])
	#print '******************************'
	X = preprocessing.scale(X_train[i])
	y = y_train[i]
	clf = SVC(kernel='poly',degree=3, verbose=True,tol=0.001)
	#clf = SVC(kernel='rbf',gamma=10)
	#clf = MLPClassifier(solver='lbfgs',alpha=1e-5,hidden_layer_sizes=(2,3), random_state=1)
	clf.fit(X, y)
	#print 'predicting', '#', i, dev
	#A = clf.predict(X_test[i])
	score = clf.score(X_test[i],y_test[i])
	#fig = figure(i,dpi=100)
	#print X_test[i][:,0]
	#print score
	scores.append(score)
#	if score == 0:
#	    print dev, i, y_test[i]
#	plot(X_test[i][:,0], scores, 'r')
#	print y_test[i]
	break
    
    print 'Done processing', dev
    return scores

def classifyForEachDay(dev, day):
    allData_ = createDataset1(dev, day)
    #visualize data
    f, (ax1, ax2) = plt.subplots(2, sharex=True)
#ax1: level vs. hr, ax2: rate vs hr    
    for each in allData_:
	levels  = [row[0] for row in each]
	mins	= [row[1] for row in each]
	rate 	= [row[2] for row in each]
	status 	= [row[3] for row in each]
	print '-----------------------------------'
	print max(rate)
	ax1.plot(mins, levels, 'g')
	ax2.plot(mins, rate, 'b--')
	x = []
	y = []
	for i in range(len(status)):
	    if status[i] :
		y.append(levels[i])
		x.append(mins[i])
	ax1.plot(x,y,'ro')
    ax1.set_xlim([0,1500])
    ax1.set_ylim([0,100])
    ax2.set_ylim([0,1])
    ax2.set_xlim([0,1500])
    #ax2.set_ylim([0,])
    ax1.set_ylabel('Battery Level')
    ax2.set_ylabel('Rate per sec')
    ax2.set_xlabel('Minutes of Day')
    
def createDataset1(dev, day):
    file1 = '/home/anudipa/Documents/Jouler_Extra/discharge/'+dev+'.p'
    file2 = '/home/anudipa/Documents/Jouler_Extra/charge/'+dev+'.p'
    try:
        dDataset = pickle.load(open(file1,'rb'))
        cDataset = pickle.load(open(file2,'rb'))
        ddata = dDataset[dev]
        cdata = cDataset[dev]
    except Exception as e:
        print 'Error', e
        return
    all_ = []
    allDates = sorted(ddata.keys())
    datesScanned = []
    for d in range(len(allDates)):
	date_ = allDates[d]
	if date_.weekday() != day:
	    continue
	datesScanned.append(date_)
	#wholeDay = [[0,i,0.0,False] for i in range(24)]  #[level, hour, rate, charge status ]
	wholeDay = []
	listOfSessions = ddata[date_]
	for i in range(len(listOfSessions)):
	    each = listOfSessions[i]
	    #entry = [[0,i,0,False] for i in range(24)]
	    lastL = -1
	    lastT = each[0][1]
	    
	    for j in range(len(each)):
		rate = 0.00
    		level = each[j][0]
		hr = each[j][1].hour
		t = each[j][1].hour*60.0+each[j][1].minute
		if level < lastL and lastT < each[j][1]:
		    time_delta = (each[j][1]-lastT).total_seconds()
		    if time_delta < 1:
			tmp = 1.00
		    else:
		    	tmp = float((lastL-level)/(each[j][1] - lastT).total_seconds())
		    if tmp > 15:
			print 'Error!', tmp, each[j], lastT, (each[j][1]-lastT).total_seconds(), (lastL-level)
		    try:
		    	rate = float("{0:.2f}".format(tmp))
		    except ValueError as e:
			print 'Error!', tmp, e
		wholeDay.append([level,t,rate,False])

		if lastL != level:
		    lastL = level
		    lastT = each[j][1]
		
	if date_ not in cdata.keys():
	    continue
	listOfSessions = cdata[date_]
	for i in range(len(listOfSessions)):
	    each = listOfSessions[i]
	    start_hr = each[0].hour
	    end_hr = each[2].hour
	    start_level = each[0]
	    delta = (each[2]-each[0]).total_seconds()/3600
	    wholeDay.append([each[1],(each[0].hour*60+each[0].minute),0.0,True])
	    wholeDay.append([each[3],(each[2].hour*60+each[2].minute),0.0,False])
	wholeDay.sort(key=lambda x: x[1])
        all_.append(wholeDay)
    print 'Number of days:', len(all_)
    #for i in range(len(all_)):
	#print 'day #', i, datesScanned[i]
	#print all_[i]
	#print '-----------------------------'
    return all_		    

#more features: level, hr, 1/2/3/4th quarter of the hour, charging or discharging, 
#pre_discharge_rate
def createDataset(dev):
    file1 = '/home/anudipa/Documents/Jouler_Extra/discharge/'+dev+'.p'
    file2 = '/home/anudipa/Documents/Jouler_Extra/charge/'+dev+'.p'
    try:
	dDataset = pickle.load(open(file1,'rb'))
	cDataset = pickle.load(open(file2,'rb'))
	ddata = dDataset[dev]
	cdata = cDataset[dev]
    except Exception as e:
	print 'Error', e
	return
#   total
#   train_num
    allData_ = [[] for i in range(7)]
    sortedD = sorted(ddata.keys())
    for d in range(len(sortedD)):
	day = sortedD[d].weekday()
	listOfSessions = ddata[sortedD[d]]
	for i in range(len(listOfSessions)):
	    each = listOfSessions[i]
	    last_ =[0,0,0,0]		#last_=[level,hr,quarter,time_now]
	    for j in range(len(each)):
		level = each[j][0]
		hr = each[j][1].hour
		mins = each[j][1].minute
		qrt = mins/15
		#print level, qrt
		if j == 0:
		    allData_[day].append([level, hr, qrt, False, 0])
		    last_ = [level, hr, qrt,each[j][1]]
		elif (each[j][1]-last_[3]).seconds > 0 and (last_[0] > level or last_[1] != hr or last_[2] != qrt):
		    if last_[0] < level:
			continue
		    #print last_,each[j]
		    rate_ = (last_[0] - level)/float(((each[j][1]-last_[3]).seconds)/60.0)      #level drop per minute
		    rate = float("{0:.2f}".format(rate_))
		    last_ = [level,hr,qrt,each[j][1]]
		    allData_[day].append([level,hr,qrt,False,rate])
    
    sortedC = sorted(cdata.keys())
    for d in range(len(sortedC)):
	day = sortedC[d].weekday()
	listOfSessions = cdata[sortedC[d]]
	if len(listOfSessions) == 0:
	    continue
        trimmedList = []
        trimmedList.append(listOfSessions[0])
        lastT = listOfSessions[0][2]
        lastL = listOfSessions[0][3]
        for i in range(1,len(listOfSessions)):
            each_ = listOfSessions[i]
            if lastL < each_[1] and lastT < each_[0]:
                trimmedList[-1][2] = each_[2]
                trimmedList[-1][3] = each_[3]
            else:
                trimmedList.append(each_)
            lastT = each_[2]
            lastL = each_[3]
        for i in range(len(trimmedList)):
            each = trimmedList[i]
	    if each[2] == -1:
		each[2] = datetime.max.time()
	    if each[3] == -1:
		each[3] = 100
	    allData_[day].append([each[1], each[0].hour, each[0].minute/15, True, 0])
	    allData_[day].append([each[3], each[2].hour, each[2].minute/15, True, 0])
    return allData_

