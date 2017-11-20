#!/usr/bin/python2

#develop neural network for routine/conservative users reading from csv
#values in csv has been extrapolated to reach level 15 to start charging
#values have been rounded to exclude too much fluctuations
#T round by 30 mins, Level binned by 5, FG by 0.1 and dLevel by 5 levels drop


import os
import numpy as np
from collections import defaultdict, Counter, OrderedDict
import pickle
from datetime import *
import csv
from pylab import *
import statsmodels as stats
import pandas as pd
import pandas.tools.plotting as pdtools
from sklearn.neural_network import MLPClassifier


#first load csv data into dataframe
#second filter out all data points for time_left_to_charge <=180 mins
#aim being we start with making the model accurate for instances close
#to charging session

def readCSV(csvF):
	datapoints = []
	with open(csvF) as csvfile:
		reader = csv.reader(csvfile, delimiter=';')
		heading = next(reader)
		for row in reader:
			time_since_start_discharge = int(row[0])
			level_now = int(row[1])
			level_drop_since_start = int(row[2])
			foreground_usage_frac = float(row[3])
			time_left_to_charge = int(row[4])
			if time_left_to_charge <= 600:
				datapoints.append([time_since_start_discharge,level_now,level_drop_since_start,foreground_usage_frac,time_left_to_charge])

	#now filter datapoints with time_left_to_charge <=180 mins
	#X are independent variables and Y is time_left_to_charge,
	#round each time_left_to_charge to multiples of 30
	print('*************', len(datapoints))
	X = []
	Y = []
	for i in range(len(datapoints)):
		X.append([datapoints[i][0], datapoints[i][1], datapoints[i][2], datapoints[i][3]])
		rem = datapoints[i][4]%60
		if (rem == 0 and datapoints[i][4] != 0) or rem==30:
			new_tl = datapoints[i][4]
		elif rem < 30:
			new_tl = int(datapoints[i][4]/60)*60+30
		else:
			new_tl = int(datapoints[i][4]/60)*60+60
		#print('rounded', datapoints[i][4], new_tl)
		Y.append(new_tl)
	dict_ = {'X':X, 'Y':Y}
	return dict_

def neural_network(X, Y):
	#now possible outputs are 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360, 390, 420, 450, 480, 510, 540, 570, 600: 20 classes
	#we will get 300 training datapoints for each of the classes
	#and keep 300 for testing set for motherload X and Y
	trainX = []
	trainY = []
	testX = []
	testY = []
	label1 = label2 = label3 = label4 = label5 = label6 =label7 =label8= 0
	label9 = label10 = label11 = label12 = label13 = label14 = label15 = 0
	label16 = label17 = label18 = label19 = label20 = 0
	for i in range(len(Y)):
		if len(testY) == 20*100 and len(trainY) == 20*400:
			print('looped till: ', i)
			break
		if Y[i] == 30:
			if label1 < 400:
				trainX.append(X[i])
				trainY.append(Y[i])
			elif 400<=label1<500:
				testX.append(X[i])
				testY.append(Y[i])
			label1 += 1
		elif Y[i] == 60:
			if label2 < 400:
				trainX.append(X[i])
				trainY.append(Y[i])
			elif 400<=label2<500:
				testX.append(X[i])
				testY.append(Y[i])
			label2 += 1
		elif Y[i] == 90:
			if label3 < 400:
				trainX.append(X[i])
				trainY.append(Y[i])
			elif 400<=label3<500:
				testX.append(X[i])
				testY.append(Y[i])
			label3 += 1
		elif Y[i] == 120:
			if label4 < 400:
				trainX.append(X[i])
				trainY.append(Y[i])
			elif 400<=label4<500:
				testX.append(X[i])
				testY.append(Y[i])
			label4 += 1
		elif Y[i] == 150:
			if label5 < 400:
				trainX.append(X[i])
				trainY.append(Y[i])
			elif 400<=label5<500:
				testY.append(Y[i])
				testX.append(X[i])
			label5 += 1
		elif Y[i] == 180:
			if label6 < 300:
				trainX.append(X[i])
				trainY.append(Y[i])
			elif 400<=label6<500:
				testX.append(X[i])
				testY.append(Y[i])
			label6 += 1
		elif Y[i] == 210:
			if label7 < 400:
				trainX.append(X[i])
				trainY.append(Y[i])
			elif 400<=label7<500:
				testX.append(X[i])
				testY.append(Y[i])
			label7 += 1
		elif Y[i] == 240:
			if label8 < 400:
				trainX.append(X[i])
				trainY.append(Y[i])
			elif 400<=label8<500:
				testX.append(X[i])
				testY.append(Y[i])
			label8 += 1
		elif Y[i] == 270:
			if label9 < 400:
				trainX.append(X[i])
				trainY.append(Y[i])
			elif 400<=label9<500:
				testX.append(X[i])
				testY.append(Y[i])
			label9 += 1
		elif Y[i] == 300:
			if label10 < 400:
				trainX.append(X[i])
				trainY.append(Y[i])
			elif 400<=label10<500:
				testX.append(X[i])
				testY.append(Y[i])
			label10 += 1
		elif Y[i] == 330:
			if label11 < 400:
				trainX.append(X[i])
				trainY.append(Y[i])
			elif 400<=label11<500:
				testX.append(X[i])
				testY.append(Y[i])
			label11 += 1
		elif Y[i] == 360:
			if label12 < 400:
				trainX.append(X[i])
				trainY.append(Y[i])
			elif 400<=label12<500:
				testX.append(X[i])
				testY.append(Y[i])
			label12 += 1
		elif Y[i] == 390:
			if label13 < 400:
				trainX.append(X[i])
				trainY.append(Y[i])
			elif 400<=label13<500:
				testX.append(X[i])
				testY.append(Y[i])
			label13 += 1
		elif Y[i] == 420:
			if label14 < 400:
				trainX.append(X[i])
				trainY.append(Y[i])
			elif 400<=label14<500:
				testX.append(X[i])
				testY.append(Y[i])
			label14 += 1
		elif Y[i] == 450:
			if label15 < 400:
				trainX.append(X[i])
				trainY.append(Y[i])
			elif 400<=label15<500:
				testX.append(X[i])
				testY.append(Y[i])
			label15 += 1
		elif Y[i] == 480:
			if label16 < 400:
				trainX.append(X[i])
				trainY.append(Y[i])
			elif 400<=label16<500:
				testX.append(X[i])
				testY.append(Y[i])
			label16 += 1
		elif Y[i] == 510:
			if label17 < 400:
				trainX.append(X[i])
				trainY.append(Y[i])
			elif 400<=label17<500:
				testX.append(X[i])
				testY.append(Y[i])
			label17 += 1
		elif Y[i] == 540:
			if label18 < 400:
				trainX.append(X[i])
				trainY.append(Y[i])
			elif 400<=label18<500:
				testX.append(X[i])
				testY.append(Y[i])
			label18 += 1
		elif Y[i] == 570:
			if label19 < 400:
				trainX.append(X[i])
				trainY.append(Y[i])
			elif 400<=label19<500:
				testX.append(X[i])
				testY.append(Y[i])
			label19 += 1
		elif Y[i] == 600:
			if label20 < 400:
				trainX.append(X[i])
				trainY.append(Y[i])
			elif 400<=label20<500:
				testX.append(X[i])
				testY.append(Y[i])
			label20 += 1

	#now the MLPClassifier model
	iter_ = 500
	m = 15
	n = 20
	print('parameters', iter_, m, n)
	clf = MLPClassifier(hidden_layer_sizes=(m,n), random_state=1, max_iter=1, warm_start=True)
	for i in range(iter_):
		clf.fit(trainX, trainY)
		#print([coef.shape for coef in clf.coefs_])
	pred_ = clf.predict(testX)
	print(len(pred_), len(testY), len(trainY), label7, label8)
	score0 = score1 = score2 = score3 = 0
	right = []
	for i in range(len(pred_)):
		if pred_[i] == testY[i]:
			score0 += 1
			#print(pred_[i], testY[i])
		elif abs(pred_[i] - testY[i]) <=30:
			score1 += 1
			#print(pred_[i], testY[i])
		elif abs(pred_[i] - testY[i]) <= 60:
			score2 += 1
		else:
			score3 += 1
		#print('predicted: ', pred_[i], ' real: ', testY[i])
	print(score0, score1, score2, score3, len(pred_))
	print('accuracy', (score0+score1)/len(pred_))
	a = 0
	b = []
	c = 0
	tl = 360
	for i in range(len(testY)):
		if testY[i] == tl:
			if abs(pred_[i] -testY[i]) <=30:
				a += 1
			else:
				b.append(abs(pred_[i]-testY[i]))
			c += 1
	print('for time left', tl, ':correct=',a,' total=', c)
	print(len(b),np.mean(b),'****************', b)
	#plotting
	fig = figure(0, dpi=100)
	err = []
	for i in range(len(pred_)):
		err.append(abs(pred_[i] - testY[i]))
	#plot(testY,err, 'ro')
	#title('Neural Network prediction with 1800 samples for <=180 mins')
	#plt.show()

if __name__ == '__main__':
	#first check reading csv and creating input/output
	dict_ = readCSV('/home/anudipa/pattern/git_scripts/usage-pattern/data/csv/0f73f649f1e0bb371f1fdcadf86f02567670315a.csv')
	print('X', len(dict_['X']))
	print('Y', dict_['Y'][:25])
	neural_network(dict_['X'], dict_['Y'])
