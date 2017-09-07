#!/usr/bin/python

import os
import numpy as np
import pandas as pd
import screenParse as sc
from collections import defaultdict, Counter, OrderedDict
import pickle
from datetime import *
from pylab import *
import matplotlib.pyplot as plt
import csv
from hmmlearn.hmm import GaussianHMM, MultinomialHMM
import seaborn as sns


#things to do:
#	load csv to a panda dataframe
#	get correlation matrix
#	PCA decomposition and plot eigen vectors
#	train hmm model and get transition matrix
#	use stochastic probability algorithm to get best possible sequence

path_to = ''
def loadCSV(dev):
	path_ = path_to +dev+'.csv'
	df = pd.read_csv(path_,sep=';', skiprows=0,header=0)
	print(df.corr)
	#plot and save correlation heatmap
	#corrmat = df.corr()
	#sns.heatmap(corrmat, vmax=1, square=False).xaxis.tick_top()
	return df

def PCA(dev):
	df = loadCSV(dev)


def getEigen(dev):
	df = loadCSV(dev)
