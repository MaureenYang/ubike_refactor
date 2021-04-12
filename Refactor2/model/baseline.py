import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pylab as plt
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
import arima

#input train_y and test_y
def HistroicalAverage(train_y, test_y):
    mean_of_train = np.mean(train_y)
    predict_y = np.full((len(test_y)), np.mean(train_y))
    RMSE = (mean_squared_error(predict_y, test_y))#, squared=False))
    return RMSE


def MovingAverage(series, n):
    predict_y = []
    test_y = []
    for i in range(0,len(series)-n):
        predict_y = predict_y + [np.average(series[i:i+n])]
        test_y = test_y + [series[n+i:n+1+i]]
        RMSE = (mean_squared_error(predict_y, test_y))#, squared=False))

    return RMSE


Estimator_Name = []
model_score_list = []

df=pd.read_csv("new_data_sno1.csv")
ndf = df[['mday','bemp']]
ndf['y']=ndf['bemp'].apply(lambda x: round(x, 0))

ndf['time'] = pd.to_datetime(ndf['mday'], format='%Y/%m/%d %H%M', errors='ignore')
ndf = ndf.drop(columns = ['bemp','mday'])
ndf = ndf.set_index(['time'])
ndf = ndf.interpolate()

#print(MovingAverage(ndf['y'],100))

arima(ndf['y'],100)
