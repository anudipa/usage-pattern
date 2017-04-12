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
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from pandas.tools.plotting import *
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf

def convert(data):
	data_type = type(data)
	if data_type == bytes : return data.decode()
	if data_type in (str,int,float): return data
	if data_type in (datetime.datetime,datetime.date): return data
	if data_type == dict: data = data.items()
	return data_type(map(convert, data))


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
	for d in sorteD:
		#print(d)
		day = d.weekday()
		wholeDay = []
		chargeDay = []
		listOfSessions = ddata[d]
		for i in range(len(listOfSessions)):
			each = listOfSessions[i]
			lastL = each[0][0]
			lastT = each[0][1]
			for j in range(len(each)):
				rate = 0.00
				level = each[j][0]
				hr = each[j][1].hour
				t = each[j][1]
				if level < lastL and lastT < t:
					time_delta = (t-lastT).total_seconds()
					if time_delta > 1:
						tmp = float((lastL-level)/(t-lastT).total_seconds())
					#if tmp > 35:
						#print("Error with timedelta",tmp, each[j],lastT, lastL)
					rate = float("{0:.4f}".format(tmp*60.0))
				wholeDay.append([day, each[j][1], level, rate, False])
				if level != lastL:
					lastL = level
					lastT = each[j][1]
		if d not in cdata.keys():
			continue
		listOfSessions = cdata[d]
		if len(listOfSessions) == 0:
			continue
		for i in range(len(listOfSessions)):
#check if there are discharge datapoints between charge sessions, discard those points
			start_ = listOfSessions[i][0]
			end_   = listOfSessions[i][2]
			if i > 0 and (start_ - endLast).total_seconds() < 300:
				wholeDay[-1][1] = end_
				wholeDay[-1][2] = listOfSessions[i][3]
				span = (end_-chargeDay[-1][1]).total_seconds()
				chargeDay[-1][3] = listOfSessions[i][3]
				chargeDay[-1][4] = span
			else:
				wholeDay.append([day,start_,listOfSessions[i][1],0.0,True])
				wholeDay.append([day,end_,listOfSessions[i][3],0.0,True])
				chargeDay.append([day,start_,listOfSessions[i][1],listOfSessions[i][3],(end_-start_).total_seconds()])

				#recreating charge samples
				if (listOfSessions[i][3] - listOfSessions[i][1]) > 5:
					print(len(wholeDay), listOfSessions[i][1], listOfSessions[i][3])
					fillInCharge(wholeDay, day, start_, end_, listOfSessions[i][1], listOfSessions[i][3])
					print(len(wholeDay))
			endLast = end_

		wholeDay.sort(key=lambda x:x[1])
		chargeDay.sort(key=lambda x:x[1])

		for i in range(len(wholeDay)):
			all_.append(wholeDay[i])

#		truth = False
#		for i in range(len(wholeDay)):
#			if truth == False:
#				all_.append(wholeDay[i])
#				truth = wholeDay[i][4]
#			else:
#				if wholeDay[i][4]== True:
#					all_.append(wholeDay[i])
#					all_[-1][4] = False
#					truth = False


		for i in range(len(chargeDay)):
			charge_.append(chargeDay[i])

	panda = DataFrame(all_, columns=['day','datetime','level','rate','status'])
	indexed_panda = panda.set_index('datetime')
	charge = DataFrame(charge_, columns=['day','datetime','start_level','end_level','span'])
	indexed_charge = charge.set_index('datetime')

	return [indexed_panda, indexed_charge]

def fillInCharge(wholeDay, day, start_time, end_time, start_level, end_level):
	span = (end_time - start_time).total_seconds()
	diff = (end_level - start_level)
	#create time series
	min_interval = int(span/diff + 1)
	freq_ = "{0:d}".format(min_interval)+'s'
	tmp = date_range(start = start_time, end = end_time, freq = freq_)
	target = start_level+1
	for i in range(1,len(tmp)-1):
		if target < end_level:
			entry = [day, tmp[i], target, 0.0, True]
			target += 1 
		else:
			entry = [day, tmp[i], target,0.0, True]
		wholeDay.append(entry)

	
 
def analysis1(panda):
#separate data in to each day and then resample data subset for 1 min
#1. get list of all unique dates
	frames = []
	uq_dates = panda.index.to_series().apply(lambda x:x.date()).unique()
	print('# of days:',len(uq_dates))
	#print(uq_dates)
	last_ = panda.index[0]
	for i in range(1,uq_dates.size):
		df = Series.to_frame(panda.ix[(panda.index.date==uq_dates[i])]['level'])
		#print(df.head())
#resample the data with 1 min freq
		df = df[~df.index.duplicated(keep='first')]
		df_ = df.resample('T').bfill()
		#df_ = df_.resample('15T').mean()
		df_ = df_.resample('15T').min()
		frames.append(df_)
		if i == 150:
			last_ = df_.index[-1]
			print(last_)
			break
	result = concat(frames)
	result.rename(columns={'':'level'}, inplace=True)
	result['level'] = result.level.astype(float)
	#print(result)
	tmp = Series.to_frame(panda.ix[(panda.index.date == uq_dates[i+1]) & (panda.index <= uq_dates[i+1]+DateOffset(hours=6))]['level'])
	#print(tmp)
	tmp = tmp[~tmp.index.duplicated(keep='first')]
	tmp = tmp.resample('T').bfill()
	out_of_sample = tmp.resample('15T').min()
	out_of_sample.rename(columns={'':'level'}, inplace=True)
	out_of_sample['level'] = out_of_sample.level.astype(float)
	print(out_of_sample.size, out_of_sample)	
	#out_of_sample = date_range(start_,end_,freq="15min")
	#result.plot()
	#lag_plot(result)
	result['first_diff'] = result.level - result.level.shift(1)
	#result['second_diff'] = result.level - result.level.shift(120)
	#result['other_diff'] = result.level - result.level.shift(12)
	#test_stationary(result.first_diff.dropna(inplace=False))
	result['seasonal_diff'] = result.level - result.level.shift(24)
	#test_stationary(result.seasonal_diff.dropna(inplace=False))
	result['seasonal_first_diff'] = result.first_diff - result.first_diff.shift(24)
	#result['seasonal_weekly_diff'] = result.seasonal_first_diff - result.seasonal_first_diff.shift(7)
	#test_stationary(result.seasonal_first_diff.dropna(inplace=False))

	#fig = plt.figure(figsize=(12,8))
	#ax1 = fig.add_subplot(211)
	#sm.graphics.tsa.plot_acf(result.first_diff.iloc[1:], lags=60, ax=ax1)
	#ax2 = fig.add_subplot(212)
	#sm.graphics.tsa.plot_pacf(result.first_diff.iloc[1:],lags=60, ax=ax2)
	#fig2 = plt.figure(figsize=(12,8))
	#ax1 = fig2.add_subplot(211)
	#sm.graphics.tsa.plot_acf(result.seasonal_first_diff.iloc[24:], lags=60, ax=ax1)
	#ax2 = fig2.add_subplot(212)
	#sm.graphics.tsa.plot_pacf(result.seasonal_first_diff.iloc[24:],lags=60, ax=ax2)

	#print(result.seasonal_first_diff)
	#lag_acf = acf(result.seasonal_first_diff.iloc[24:], nlags=30)
	#lag_pacf = pacf(result.seasonal_first_diff.iloc[24:], nlags=30)
	#fig = plt.figure(figsize=(12,8))
	#ax1 = fig.add_subplot(211)
	mod = sm.tsa.ARMA(result.level,(1,3))
	#mod = stats.tsa.arima_model.ARIMA(result.level,order=(0,1,2))
	#mod = stats.tsa.statespace.sarimax.SARIMAX(result.level, trend='n', order=(0,1,1), seasonal=(0,1,1,24), simple_differencing=True)
	#res = mod.fit(trend='c',disp=False)
	res = mod.fit()
#	print(res.summary())
	print(sm.stats.durbin_watson(res.resid.values))
	predict_ = res.predict(start='2015-08-14')
	#predict_ = res.predict(out_of_sample)
	print(predict_)
	#fig = plt.figure(figsize=(12,8))
	#ax = fig.add_subplot(111)
	#res.resid.plot(ax=ax)
	resid = res.resid
	forecast_ = res.forecast(out_of_sample.size)[0]
	#fig1=plt.figure(figsize=(12,8))
	#ax = fig1.add_subplot(111)
	#sm.graphics.qqplot(resid,line='q',ax=ax,fit=True)
	#fig2 = plt.figure(figsize=(12,8))
	#ax1 = fig2.add_subplot(211)
	#sm.graphics.tsa.plot_acf(resid.values.squeeze(), lags=50, ax=ax1)
	#ax2 = fig2.add_subplot(212)
	#sm.graphics.tsa.plot_pacf(resid, lags=50, ax=ax2)
	#fig, ax = plt.subplots(figsize=(12, 8))
	#ax = result.level.ix['07-2015':].plot(ax=ax)
	#res.plot_predict('06-01-2015', exog=np.array(range(1,4*24*7)), plot_insample=True)
	print(mean_forecast_err(out_of_sample.level,forecast_))
	d_ = date_range(out_of_sample.index[0],periods=out_of_sample.size, freq="15min")
	f_ = Series(forecast_, index=d_)
	out_of_sample['forecast'] = f_
	#out_of_sample[['level','forecast']].plot(figsize=(12,8))
	#print(result.tail(24))
	print(out_of_sample)
	#print(result.level.tail(100))
	#result['forecast'] = res.predict(result.level.tail(24*7))
	#result[['level','forecast']].plot(figsize=(12,8))
	print(result.tail(24*7))



def mean_forecast_err(y, yhat):
    return y.sub(yhat).mean()


#http://www.seanabu.com/2016/03/22/time-series-seasonal-ARIMA-model-in-python/
def test_stationary(timeseries):
	#determine rolling statistics
	rolmean = timeseries.rolling(center=False, window=60).mean() 
	rolstd = timeseries.rolling(center=False, window=60).std()

	#plot rolling statistics
	fig = plt.figure(figsize=(15,10))
	orig = plt.plot(timeseries, color='blue', label='Original')
	mean = plt.plot(rolmean, color='red', label='Rolling Mean')
	std = plt.plot(rolstd, color='black', label='Rolling Std')
	plt.legend(loc='best')
	plt.title('Rolling Mean & Standard Deviation')
	plt.show()

	#perform Dickey-Fuller test
	print('Dickey-Fuller test results:')
	dftest = adfuller(timeseries, autolag='AIC')
	dfoutput = Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
	for key, value in dftest[4].items():
		dfoutput['Critical Value (%s)'%key] = value
	print(dfoutput)
