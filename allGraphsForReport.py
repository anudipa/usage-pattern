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
from scipy.stats import itemfreq
import timeit
import testCases
import dischargingRate as dr


def getDataDumpForAll():
	all_data = dr.doInPool(132)
	return all_data


def graphDischargeSpan():
	all_data = getDataDumpForAll()
	result = []
	for i in range(len(all_data)):
		dev = next(iter(all_data[i]))
		#if dev == '2927a38a4daa0872371e822effc1499098f5fd9b':
		#	continue
		data_ = all_data[i][dev]
		x = []
		for each in data_.keys():
			slist = sorted(data_[each], key = lambda x: x[0])
			span = (slist[-1][0] - slist[0][0]).total_seconds()/60
			drop = slist[0][1] - slist[-1][1]
			#x.append(round(span,3))
			if (span < 5 and drop < 2) or span > 10000:
				print(dev, '---', round(span, 2), drop)
				continue
			x.append(round(span,2))
			
		result.append(x)
	slist = sorted(result, key=lambda x: np.mean(x))
	fig, ax = plt.subplots()
	bp = ax.boxplot(slist, sym='',whis=[25,75],patch_artist = True)
	for box in bp['boxes']:
		box.set(color='#FFFFFF', linewidth=1)
		box.set(facecolor='#c0c0c0')
	for median in bp['medians']:
		median.set(color='#FF0000', linewidth=3)
	ax.tick_params(labelbottom='off')
	ax.set_ylabel('Discharging span in minutes')
	ax.set_xlabel('Users')
	fig.show()

def loadDeviceOne(dev):
	filename = '/home/anudipa/Documents/Jouler_Extra/final/shortlisted/'+dev+'.p'
	try:
		dDataset = pickle.load(open(filename,'rb'), encoding='bytes')
		device = next(iter(dDataset))
		
		print(dDataset[dev].keys())	
	except Exception as e:
		print('Error',e)
