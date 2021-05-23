# -*- coding: utf-8 -*-
"""
Created on Tue May 11 00:03:04 2021

@author: Maureen
"""
import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt                  # plots
import seaborn as sns                            # more plots
import data_preprocessor as dp
import warnings                                  # `do not disturbe` mode
warnings.filterwarnings('ignore')
import statsmodels.formula.api as smf            # statistics and econometrics
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs

from sklearn.metrics import mean_squared_error, mean_absolute_error

import statsmodels.tsa.stattools as ts
from itertools import product                    # some useful functions

#%%
filepath = "E:/csvfile/source/"
filesinpath = os.listdir(filepath)
for f in sorted(filesinpath): #for each file, ran model
    print("file name:", f)
    df = pd.read_csv(filepath + f)
    df = df[df.sno != 'sno']
    df = df.drop_duplicates(keep='first')
    l = len(df)  
    df['time'] =  pd.date_range("2018-02-01", periods=l, freq="H")
    #df['time'] = pd.to_datetime(df['Unnamed: 0'], format='%Y/%m/%d %H%M%S', errors='ignore') 
    df = df.set_index(pd.DatetimeIndex(df['time']))
    df = df.sort_index()
    df=df.drop(columns=['Unnamed: 0']).asfreq('H')
    break


#%%
ny = dp.data_preprocess(df)
ny = ny['20180301':'20180630']
small_ny = ny['20180601':'20180630']
ts = ny.sbi
plt.figure(figsize=(15, 7))
plt.plot(small_ny.bemp)
plt.title('2018 Bike Number')
plt.grid(True)
plt.show()

#%%
plt.figure(figsize=(15, 7))
plt.plot(ny.sbi)
plt.title('2018 Bike Number')
plt.grid(True)
plt.show()

#%%
def plotMovingAverage(series, window, plot_intervals=False, scale=1.96, plot_anomalies=False):

    """
        series - dataframe with timeseries
        window - rolling window size 
        plot_intervals - show confidence intervals
        plot_anomalies - show anomalies 
    """
    rolling_mean = series.rolling(window=window).mean()

    plt.figure(figsize=(15,5))
    plt.title("Moving average\n window size = {}".format(window))
    plt.plot(rolling_mean, "g", label="Rolling mean trend")

    rmse = np.sqrt(mean_squared_error(series[window:], rolling_mean[window:]))

    # Plot confidence intervals for smoothed values
    if plot_intervals:
        mae = mean_absolute_error(series[window:], rolling_mean[window:])
        deviation = np.std(series[window:] - rolling_mean[window:])
        lower_bond = rolling_mean - (mae + scale * deviation)
        upper_bond = rolling_mean + (mae + scale * deviation)
        plt.plot(upper_bond, "r--", label="Upper Bond / Lower Bond")
        plt.plot(lower_bond, "r--")
        
        # Having the intervals, find abnormal values
        if plot_anomalies:
            anomalies = pd.DataFrame(index=series.index, columns=series.columns)
            anomalies[series<lower_bond] = series[series<lower_bond]
            anomalies[series>upper_bond] = series[series>upper_bond]
            plt.plot(anomalies, "ro", markersize=10)
        
    plt.plot(series[window:], label="Actual values")
    plt.legend(loc="upper left")
    plt.grid(True)
    
    return rmse

#%%
ma_rms12 = plotMovingAverage(ts, 12, plot_intervals=True)
ma_rmse6 = plotMovingAverage(ts, 6, plot_intervals=True)
ma_rmse3 = plotMovingAverage(ts, 3, plot_intervals=True)

print('ma_rms12:',ma_rms12)
print('ma_rmse6:',ma_rmse6)
print('ma_rmse3:',ma_rmse3)





#%%

def plotHistoricalAverage(series, window, plot_intervals=False, scale=1.96, plot_anomalies=False):

    """
        series - dataframe with timeseries
        window - rolling window size 
        plot_intervals - show confidence intervals
        plot_anomalies - show anomalies 
    """
    
    plt.figure(figsize=(15,5))
    plt.title("Historical Average")
    remain_size = len(series) - window
    historical_mean = pd.Series([series[:window].mean()] * remain_size,index=series[window:].index)
    plt.plot(historical_mean, "g", label="Historical Mean")
    rmse = np.sqrt(mean_squared_error(series[window:], historical_mean))

    # Plot confidence intervals for smoothed values
    if plot_intervals:
        mae = mean_absolute_error(series[window:], historical_mean)
        deviation = np.std(series[window:]- historical_mean)
        lower_bond = historical_mean - (mae + scale * deviation)
        upper_bond = historical_mean + (mae + scale * deviation)
        plt.plot(upper_bond, "r--", label="Upper Bond / Lower Bond")
        plt.plot(lower_bond, "r--")
        
        # Having the intervals, find abnormal values
        if plot_anomalies:
            anomalies = pd.DataFrame(index=series.index, columns=series.columns)
            anomalies[series<lower_bond] = series[series<lower_bond]
            anomalies[series>upper_bond] = series[series>upper_bond]
            plt.plot(anomalies, "ro", markersize=10)
        
    plt.plot(series, label="Actual values")
    plt.legend(loc="upper left")
    plt.grid(True)
    
    return rmse

win = len(ts['20180301':'20180531'])
ha_rmse = plotHistoricalAverage(ts,win,plot_intervals=True)
print('ha_rmse:',ha_rmse)

#%%

def tsplot(y, lags=None, figsize=(12, 7), style='bmh'):
    """
        Plot time series, its ACF and PACF, calculate Dickey–Fuller test
        
        y - timeseries
        lags - how many lags to include in ACF, PACF calculation
    """
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
        
    with plt.style.context(style):    
        fig = plt.figure(figsize=figsize)
        layout = (2, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))
        y.plot(ax=ts_ax)
        result = sm.tsa.stattools.adfuller(y)
        print(result)
        p_value = result[1]
        ts_ax.set_title('Time Series Analysis Plots\n Dickey-Fuller: p={0:.5f}'.format(p_value))
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)
        plt.tight_layout()
        
        
#%%
def optimizeSARIMA(ts, parameters_list, d, D, s):
    """
        Return dataframe with parameters and corresponding AIC
        
        parameters_list - list with (p, q, P, Q) tuples
        d - integration order in ARIMA model
        D - seasonal integration order 
        s - length of season
    """
    
    results = []
    best_aic = float("inf")

    for param in parameters_list:
        print('param:',param)
        # we need try-except because on some combinations model fails to converge
        try:
            model=sm.tsa.statespace.SARIMAX(ts.values, order=(param[0], d, param[1]), 
                                            seasonal_order=(param[2], D, param[3], s)).fit(low_memory=True)
        except:
            continue
            
        aic = model.aic
        # saving best model, AIC and parameters
        if aic < best_aic:
            best_model = model
            best_aic = aic
            best_param = param
        results.append([param, model.aic])

    result_table = pd.DataFrame(results)
    result_table.columns = ['parameters', 'aic']
    # sorting in ascending order, the lower AIC is - the better
    result_table = result_table.sort_values(by='aic', ascending=True).reset_index(drop=True)
    
    return result_table
#%%
train_ts = ts['20180301':'20180531']
#%%
tsplot(ts,lags=60)
#%%
ts_diff = ts - ts.shift(24)
tsplot(ts_diff[24+1:], lags=60)
#%%
ps = range(2, 4)
d = 0
qs = range(2, 4)
Ps = range(0, 2)
D = 1
Qs = range(0, 2)
s = 24 # season length is still 24

parameters = product(ps, qs, Ps, Qs)
parameters_list = list(parameters)
print(len(parameters_list))
result_table = optimizeSARIMA(train_ts, parameters_list, d, D, s)
#%%
#print(result_table)
#p, q, P, Q = result_table.parameters[0]
# SARIMAX(3, 1, 2)x(3, 1, [1], 24)
# SARIMAX(6, 1, 0)x(0, 1, [1], 24) :
p = 6
d = 1
q = 0
P = 0
D = 1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
Q = 1
s = 24

train_ts = train_ts.asfreq('H')
best_model=sm.tsa.statespace.SARIMAX(train_ts,order=(p, d, q),trend='c',seasonal_order=(P, D, Q, s))
best_fit = best_model.fit()
print(best_fit.summary())
#%%
def plotSARIMA(series, window, model, plot_intervals=False, scale=1.96, plot_anomalies=False):

    """
        series - dataframe with timeseries
        window - rolling window size 
        plot_intervals - show confidence intervals
        plot_anomalies - show anomalies 
    """
    
    plt.figure(figsize=(15,5))
    plt.title("SRAIMA")
    remain_size = len(series) - window
    prediction = model.predict(start=window, end=len(series)-1)
    pred_s = pd.DataFrame(prediction, index = series[window:].index)
    
    plt.plot(pred_s, "r", label="Prediction")
    
    rmse = np.sqrt(mean_squared_error(series[window:], prediction))

    # Plot confidence intervals for smoothed values
    if plot_intervals:
        mae = mean_absolute_error(series[window:], prediction) # truth and prediction
        deviation = np.std(series[window:]- prediction)
        lower_bond = prediction - (mae + scale * deviation)
        lower_bond_df = pd.DataFrame(lower_bond, index = series[window:].index)
        upper_bond = prediction + (mae + scale * deviation)
        upper_bond_df = pd.DataFrame(upper_bond, index = series[window:].index)
        plt.plot(upper_bond_df, "r--", label="Upper Bond / Lower Bond")
        plt.plot(lower_bond_df, "r--")
        
        # Having the intervals, find abnormal values
        if plot_anomalies:
            anomalies = pd.DataFrame(index=series.index, columns=series.columns)
            anomalies[series<lower_bond] = series[series<lower_bond]
            anomalies[series>upper_bond] = series[series>upper_bond]
            plt.plot(anomalies, "ro", markersize=10)
            
        
    plt.plot(series, label="Actual values")
    plt.legend(loc="upper left")
    plt.grid(True)
    
    return rmse


#%%
win = len(ts['20180301':'20180531'])
sarima_rmse = plotSARIMA(ts,win,best_fit,scale=1.96)
print('sarima rmse:',sarima_rmse)
#%%
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(ts.dropna(),freq=24)
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid
decomposition
plt.figure(figsize=[100, 7])
decomposition.plot()
#%%
#train_ts = ts['20180301':'20180531']
p = 6
d = 1
q = 0
P = 0
D = 1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
Q = 1
s = 24

import datetime as dt

start_date = dt.datetime.strptime("20180501 00:00:00", "%Y%m%d %H:%M:%S")
end_date = dt.datetime.strptime("20180530 23:00:00", "%Y%m%d %H:%M:%S")

tdelta = dt.timedelta(hours=5)

pred_start_date = dt.datetime.strptime("20180601 00:00:00", "%Y%m%d %H:%M:%S")
pred_end_date = pred_start_date + tdelta


pred = pd.DataFrame()
for pred_date in pd.date_range(start='20180601 00:00:00', end='20180630 23:00:00',freq='H'):
    
    pred_date = enddate + dt.timedelta(hours=1) 
    pred_date_str = pred_date.strftime("%Y%m%d %H:%M:%S")
    
    pred_end_date = pred_date + dt.timedelta(hours=5) 
    pred_end_date_str = pred_end_date.strftime("%Y%m%d %H:%M:%S")
    print(startdate_str, ',',enddate_str,',',pred_date_str)
    
    '''
    train_ts = ts[startdate_str:enddate_str]
    train_ts = train_ts.asfreq('H')
    best_model=sm.tsa.statespace.SARIMAX(train_ts,order=(p, d, q),trend='c',seasonal_order=(P, D, Q, s))
    best_fit = best_model.fit(disp=-1)
    prediction = best_fit.predict(start=pred_date_str, end=pred_end_date_str)
    pred = pd.concat([pred, prediction])
    print(prediction)
    
    '''
    startdate = startdate + dt.timedelta(hours=5) 
    startdate_str = startdate.strftime("%Y%m%d %H:%M:%S")
    
    enddate = enddate + dt.timedelta(hours=5) 
    enddate_str = enddate.strftime("%Y%m%d %H:%M:%S")
    

#%%
for pred_date in pd.date_range(start='20180601 00:00:00', end='20180630 23:00:00',freq='3H'):
    
    end_date = pred_date - dt.timedelta(hours=1)
    start_date = pred_date - dt.timedelta(days=30)
    pred_end_date = pred_date + dt.timedelta(hours=2)
    
    start_date_str = start_date.strftime("%Y%m%d %H:%M:%S")
    end_date_str = end_date.strftime("%Y%m%d %H:%M:%S")
    pred_date_str = pred_date.strftime("%Y%m%d %H:%M:%S")
    pred_end_date_str = pred_end_date.strftime("%Y%m%d %H:%M:%S")
    #print(start_date_str, ',',end_date_str,',',pred_date_str,',',pred_end_date_str)
    
    train_ts = ts[start_date_str:end_date_str]
    train_ts = train_ts.asfreq('H')
    best_model=sm.tsa.statespace.SARIMAX(train_ts,order=(p, d, q),trend='c',seasonal_order=(P, D, Q, s))
    best_fit = best_model.fit(disp=-1)
    prediction = best_fit.predict(start=pred_date_str, end=pred_end_date_str)
    pred = pd.concat([pred, prediction])

    
#%%
nts = ts[pred.index]

if True:
    plt.figure(figsize=(15,5))
    plt.title("SRAIMA")
    
    pred_s = pred
    plt.plot(pred_s, "r", label="Prediction")
    
    rmse = np.sqrt(mean_squared_error(nts, pred_s))
    print(rmse)
    # Plot confidence intervals for smoothed values
    if False:
        mae = mean_absolute_error(ts[window:], prediction) # truth and prediction
        deviation = np.std(ts[window:]- prediction)
        lower_bond = prediction - (mae + scale * deviation)
        lower_bond_df = pd.DataFrame(lower_bond, index = ts[window:].index)
        upper_bond = prediction + (mae + scale * deviation)
        upper_bond_df = pd.DataFrame(upper_bond, index = ts[window:].index)
        plt.plot(upper_bond_df, "r--", label="Upper Bond / Lower Bond")
        plt.plot(lower_bond_df, "r--")
        
        # Having the intervals, find abnormal values
        if False:
            anomalies = pd.DataFrame(index=ts.index, columns=ts.columns)
            anomalies[ts<lower_bond] = ts[ts<lower_bond]
            anomalies[ts>upper_bond] = ts[ts>upper_bond]
            plt.plot(anomalies, "ro", markersize=10)
            
        
    plt.plot(ts['20180301':'20180630'], label="Actual values")
    plt.legend(loc="upper left")
    plt.grid(True)
    

