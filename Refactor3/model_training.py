# -*- coding: utf-8 -*-
"""
Created on Tue May 11 00:03:04 2021

@author: Maureen
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import data_preprocessor as dp

import statsmodels.tsa.api as smt
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error
from arch.unitroot import ADF
import statsmodels.tsa.stattools as ts
from itertools import product
import datetime as dt


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
        result = ADF(y)
        print(result)

        ts_ax.set_title('Time Series Analysis Plots\n Dickey-Fuller: p={0:.5f}'.format(p_value))
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)
        plt.tight_layout()


def optimizeSARIMA(series, parameters_list, s):
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
            model=sm.tsa.statespace.SARIMAX(series.values, order=(param[0], param[1], param[2]),
                                            seasonal_order=(param[3], param[4], param[5], s)).fit(low_memory=True)
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


    plt.plot(series[window:], label="Actual values")
    plt.legend(loc="upper left")
    plt.grid(True)

    return rmse

def plot_prediction(title_str, series, startd, endd, pred, plot_intervals=False, scale=1.96, plot_anomalies=False, fig_sz=(17,7)):

    plt.figure(figsize=fig_sz)
    plt.title(title_str)
    print(len(pred))
    nts = series[pred.index]
    print(len(nts))
    rmse = np.sqrt(mean_squared_error(nts, pred))
    print(rmse)
    if plot_intervals:
        mae = mean_absolute_error(nts, pred) # truth and prediction
        deviation = np.std(nts- pred)
        lower_bond = pred - (mae + scale * deviation)
        lower_bond_df = pd.DataFrame(lower_bond, index = nts.index)
        upper_bond = pred + (mae + scale * deviation)
        upper_bond_df = pd.DataFrame(upper_bond, index = nts.index)
        plt.plot(upper_bond_df, "r--", label="Upper Bond / Lower Bond")
        plt.plot(lower_bond_df, "r--")

        # Having the intervals, find abnormal values
        if plot_anomalies:
            anomalies = pd.DataFrame(index=series.index, columns=series.columns)
            anomalies[series<lower_bond] = series[series<lower_bond]
            anomalies[series>upper_bond] = series[series>upper_bond]
            plt.plot(anomalies, "ro", markersize=10)

    plt.plot(pred, "r", label="Prediction")
    plt.plot(nts, label="Actual values")
    plt.legend(loc="upper left")
    plt.grid(True)

    return rmse


def predict_SARIMA(series, params = [(6,1,0),(0,1,1,24)], startd='20180601 00:00:00', endd='20180630 23:00:00', freq=1):

    pred = pd.DataFrame()
    try:
        f_str = str(freq)+'H'
        for pred_date in pd.date_range(start=startd, end=endd,freq = f_str):
            print('predict_SARIMA:', pred_date, 'freq:',f_str)
            end_date = pred_date - dt.timedelta(hours=1)
            start_date = pred_date - dt.timedelta(days=15)
            pred_end_date = pred_date + dt.timedelta(hours=(freq-1))

            start_date_str = start_date.strftime("%Y%m%d %H:%M:%S")
            end_date_str = end_date.strftime("%Y%m%d %H:%M:%S")
            pred_date_str = pred_date.strftime("%Y%m%d %H:%M:%S")
            pred_end_date_str = pred_end_date.strftime("%Y%m%d %H:%M:%S")
            #print(start_date_str, ',',end_date_str,',',pred_date_str,',',pred_end_date_str)

            train_ts = series[start_date_str:end_date_str]
            series = series.asfreq('H')
            best_model=sm.tsa.statespace.SARIMAX(train_ts,order=params[0],trend='c',seasonal_order=params[1])
            best_fit = best_model.fit(disp=-1)
            prediction = best_fit.predict(start=pred_date_str, end=pred_end_date_str)
            pred = pd.concat([pred, prediction])
    except Exception as e:
        print('Predict_SARIMA Error:', e)

    return pred

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
    df['time'] = pd.to_datetime(df['Unnamed: 0'], format='%Y/%m/%d %H%M%S', errors='ignore')
    df = df.set_index(pd.DatetimeIndex(df['time']))
    df = df.sort_index()
    df=df.drop(columns=['Unnamed: 0']).asfreq('H')
    break
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
    df = df.set_index(pd.DatetimeIndex(df['time'])).sort_index()
    df = df.drop(columns=['Unnamed: 0']).asfreq('H')
    df = dp.data_preprocess(df)
    break
#%%

ny = df['20180201':'20181230']
res_ny = ny['20180301':'20180331']
ts_data = ny.sbi
#ts_data = ny.bemp

#%%
#shows the time series
plt.figure(figsize=(15, 7))
plt.plot(res_ny.bemp)
plt.title('2018 Empty Number')
plt.grid(True)
plt.show()

plt.figure(figsize=(15, 7))
plt.plot(res_ny.sbi)
plt.title('2018 Bike Number')
plt.grid(True)
plt.show()


#%%

train_start_date = '20180301 00:00:00'
train_end_date = '20180930 23:00:00'
test_start_date = '20181001 00:00:00'
test_end_date = '20181230 23:00:00'

train_ts = ts_data[train_start_date: train_end_date]
test_y = ts_data[test_start_date: test_end_date]

#%%
# Moving Average

'''
ma_rms12 = plotMovingAverage(ts_data, 12, plot_intervals=True,plot_anomalies=True)
ma_rmse6 = plotMovingAverage(ts_data, 6, plot_intervals=True,plot_anomalies=True)
ma_rmse3 = plotMovingAverage(ts_data, 3, plot_intervals=True,plot_anomalies=True)
ma_rmse1 = plotMovingAverage(ts_data, 1, plot_intervals=True,plot_anomalies=True)

print('ma_rms12:',ma_rms12)
print('ma_rmse6:',ma_rmse6)
print('ma_rmse3:',ma_rmse3)
print('ma_rmse1:',ma_rmse1)
'''
for w in [3,6,12]:
    start_date = (dt.datetime.strptime(test_start_date,"%Y%m%d %H:%M:%S") - dt.timedelta(hours=(w))).strftime("%Y%m%d %H:%M:%S")
    ma_y = ts_data[start_date:test_end_date]
    rolling_mean = ma_y.rolling(window=w).mean().shift(1).dropna()
    rmse = plot_prediction("Moving Average window size({})".format(w), ts_data, train_start_date, train_end_date, rolling_mean)
    print('Moving Average window({}), rmse:{}'.format(w, rmse))

#%%
#Histroical Average
'''
win = len(ts['20180301':'20180531'])
ha_rmse = plotHistoricalAverage(ts,win,plot_intervals=True)
print('ha_rmse:',ha_rmse)
'''

remain_size = len(test_y)
historical_mean = pd.Series([train_ts.mean()] * remain_size,index=test_y.index)
rmse = plot_prediction("Historical Average",ts_data, train_start_date, train_end_date, historical_mean)
print('Historical Average, rmse:{}'.format(rmse))
#%%
#print acf and pacf
tsplot(ts_data,lags=60)
ts_diff = ts_data - ts_data.shift(1)
tsplot(ts_diff[1+1:], lags=60)
ts_diff2 = ts_diff - ts_diff.shift(24)
tsplot(ts_diff2[24+2:], lags=60)

#%%
ps = range(2, 5)
d =  range(0, 2)
qs = range(0, 3)
Ps = range(0, 3)
D =  range(0, 2)
Qs = range(0, 3)
s = 24 # season length is still 24

parameters = product(ps,d,qs, Ps,d,Qs)
parameters_list = list(parameters)
print(len(parameters_list))
result_table = optimizeSARIMA(train_ts, parameters_list,s)

  #%%
#print(result_table)
model_list = []
fit_list = []
for i in [0]:
    print('param:',result_table.parameters[i])
    p,d,q, P, D,Q = result_table.parameters[i]
    # SARIMAX(3, 1, 2)x(3, 1, [1], 24)
    # SARIMAX(6, 1, 0)x(0, 1, [1], 24)
    '''
    p = 6
    d = 1
    q = 0
    P = 0
    D = 1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
    Q = 1
    '''
    s = 24

    train_ts = train_ts.asfreq('H')

    best_model = sm.tsa.statespace.SARIMAX(train_ts,order=(p, d, q),trend='c',seasonal_order=(P, D, Q, s))
    best_fit = best_model.fit(disp=-1)

    model_list = model_list + [best_model]
    fit_list = fit_list + [best_fit]
    #print(best_fit.summary())

#%%
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(ts_diff2[test_start_date:test_end_date].dropna(),freq=24)
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid
decomposition
plt.figure(figsize=[100, 7])
decomposition.plot()

#%%
#arima_params=[(2,0,2),(1,0,1,24)]
arima_params=[(2,0,1),(1,1,2,24)]
pred_list = []
rmse_list = []
ts_data = ts_data.asfreq('H')
for i in [1]:
    pred = predict_SARIMA(ts_data, startd =test_start_date,endd=test_end_date, freq=1)
    pred_list = pred_list + [pred]
    sarima_title = "SARIMA ({},{},{}) ({},{},{},{}), freq={}h Prediction".format(arima_params[0][0],arima_params[0][1],arima_params[0][2],
                                                                                 arima_params[1][0],arima_params[1][1],arima_params[1][2],arima_params[1][3],1)
#%%
    #print(ts_data[ts_data.isna()])
    #pred = pred.interpolate()
    #print(pred['20181009':'20181009'].isna())
    #pred['20181009':'20181009'][pred['20181009':'20181009'] < 0] = None
    #pred['20181009':'20181009'].plot()
    #nts = ts_data[pred.index].plot()
    #pred['20181009 22:00:00'] = np.nan
    pred['20181009':'20181009'].plot()
    #pred = pred.interpolate()
    #print(pred['20181009':'20181009'][pred['20181009':'20181009'] < 0])

    #pred.plot()
    #rmse = np.sqrt(mean_squared_error(nts, pred))
    #print(rmse)
#%%
    rmse = plot_prediction(sarima_title, ts_data, train_start_date, train_end_date, pred)
    rmse_list = rmse_list + [rmse]
    print('SARIMA p({}), rmse:{}'.format(i,rmse))

print(rmse_list)

#%%
print(result_table)
result_table.to_csv('arima_result.csv')
#%%
for i in [0,1,2,3,4]:
    best_fit = fit_list[i]
    print(best_fit.summary())

   #%%
for i in [0,1,2,4]:
    best_fit = fit_list[i]
    w = len(ts_data[train_start_date:train_end_date])
    rmse = plotSARIMA(ts_data, w, best_fit)
    rmse_list = rmse_list + [rmse]
    print('SARIMA p({}), rmse:{}'.format(i,rmse))

print(rmse_list)
#%%
p = 2
d = 0
q = 1
P = 1
D = 1
Q = 2
s = 24
train_ts = train_ts.asfreq('H')
best_model = sm.tsa.statespace.SARIMAX(train_ts,order=(p, d, q),trend='c',seasonal_order=(P, D, Q, s))
best_fit = best_model.fit(disp=-1)
w = len(ts_data['20180301':'20180531'])
#%%
rmse = plotSARIMA(ts_data, w, best_fit)
print(rmse)
