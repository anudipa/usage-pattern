import os
from pandas import *
import numpy as np
from multiprocessing import Process, current_process, Pool
import pickle
import json
from datetime import *
from itertools import cycle
from collections import Counter, defaultdict
from pylab import *
import statsmodels as stats


def convert(data):
        data_type = type(data)
        if data_type == bytes : return data.decode()
        if data_type in (str,int,float): return data
        if data_type in (datetime.datetime,datetime.date): return data
        if data_type == dict: data = data.items()
        return data_type(map(convert, data))

#load device data in the right format
def loadDevice(dev):
        file1 = '/home/anudipa/Documents/Jouler_Extra/discharge2/'+dev+'.p'
        file2 = '/home/anudipa/Documents/Jouler_Extra/charge2/'+dev+'.p'
        print(file1,file2)
        try:
                print('Starting!!!')
                tmp1 = pickle.load(open(file1,'rb'), encoding='bytes')
                tmp2 = pickle.load(open(file2,'rb'), encoding='bytes')
                dDataset = convert(tmp1)
                cDataset = convert(tmp2)
                ddata = dDataset[dev]
                cdata = cDataset[dev]
        except Exception as e:
                print(e)
                return
#panda discharge dataframe : day, datetime, level, rate, status (charge/discharge)
#panda charge dataframe    : day, datetime, start level, end level, span
        all_ = []
        charge_ = []
        ignored_ = []
        #print('#############################')
        #print(len(ddata.keys()))
        sorteD = sorted(ddata.keys())

