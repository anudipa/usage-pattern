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


#for each device plot time vs battery level for every session

def plotOne(dict_, dev):
	fig, ax = plt.subplots()
	ax.set_title(dev)
	ax.set_xlabel('Time of day')
	ax.set_ylabel('Battery level')
	ax.set_ylim(0,100)
	ax.set_xlim(slist[0].date, slist[-1].date)
	slist = sorted(dict_.keys())
	m = slist[0].month
	for i in range(len(slist)):
		if m != sli
	
