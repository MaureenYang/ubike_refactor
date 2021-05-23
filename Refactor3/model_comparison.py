import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os, sys, time

import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

from sklearn.metrics import mean_squared_error
import data_preprocessor as dp

# model
import baseline
from ridge_model import ridge
from lasso_model import lasso
from gb_model import gb
from ada_model import ada
from rf_model import rf
from xgb_model import xgb


'''
    * each model for each station
'''
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    
''' configuration '''
small_set_flag = True

target_tag = ['sbi']
estimator_list = [lasso, ridge, gb, rf, ada, xgb]
title_list = ['Lasso ','Ridge ','Gradient Boost ','Random Forest ','Adaboost ','XGBoost']

estimator_list = [lasso, ridge, rf, ada, xgb]
title_list = ['Lasso ','Ridge ','Random Forest ','Adaboost ','XGBoost']
#estimator_list = [ada, xgb]
#title_list = ['Adaboost ', 'XGBoost ']
station_list = [1] #range(1, 100, 40)


''' const'''
filepath = "E:/csvfile/source2/"
#drop features
feature_sbi_list = ['sbi','sbi_1h', 'sbi_2h', 'sbi_3h','sbi_4h', 'sbi_5h', 'sbi_6h','sbi_7h', 'sbi_8h', 'sbi_9h','sbi_10h', 'sbi_11h', 'sbi_12h', 'sbi_1d', 'sbi_2d', 'sbi_3d', 'sbi_4d', 'sbi_5d', 'sbi_6d', 'sbi_7d']
feature_bemp_list = ['bemp','bemp_1h', 'bemp_2h', 'bemp_3h','bemp_4h', 'bemp_5h', 'bemp_6h','bemp_7h', 'bemp_8h', 'bemp_9h','bemp_10h', 'bemp_11h', 'bemp_12h', 'bemp_1d', 'bemp_2d', 'bemp_3d', 'bemp_4d', 'bemp_5d', 'bemp_6d', 'bemp_7d']
feature_weather_list = ['HUMD', 'H_24R', 'PRES', 'TEMP', 'WDSE', 'PrecpHour', 'UVI', 'Visb', 'WD_E', 'WD_ENE', 'WD_ESE', 'WD_N', 'WD_NE', 'WD_NNE', 'WD_NNW', 'WD_NW', 'WD_S', 'WD_SE', 'WD_SSE', 'WD_SSW', 'WD_SW', 'WD_W', 'WD_WNW', 'WD_WSW']
feature_ans = ['y_bemp', 'y_sbi']
feature_station = ['sno','lat','lng']


''' functions '''    
def plot_prediction(title_str, series, startd, endd, pred, plot_intervals=False, scale=1.96, plot_anomalies=False, fig_sz=(17,7)):
    
    plt.figure(figsize=fig_sz)
    plt.title(title_str)
    
    nts = series[pred.index]
    rmse = np.sqrt(mean_squared_error(nts, pred))
    
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


#%%
df_list =[]
mse_list = []
result_list = []
#read data from file
for sno in station_list:
    f = 'data_sno_'+str(sno).zfill(3)+'.csv'
    print("file name:", f)
    df = pd.read_csv(filepath + f)
    new_df = dp.data_preprocess(df).dropna()

#%%
# feature handling
    for tag in target_tag:
        if tag == 'sbi':
            final_drop_list = feature_ans + feature_bemp_list + feature_station #+ feature_weather_list
            X = new_df.drop(columns = final_drop_list)
            Y = new_df[['y_sbi']]
    
        if tag == 'bemp':
            final_drop_list = feature_ans + feature_sbi_list + feature_station #+ feature_weather_list
            X = new_df.drop(columns = final_drop_list)
            Y = new_df[['y_bemp']]
            
#%%
        #split data
        if small_set_flag:
            train_start_date = '20180301 00:00:00'
            train_end_date = '20180531 23:00:00'
            test_start_date = '20180601 00:00:00'
            test_end_date = '20180630 23:00:00'
        else:
            train_start_date = '20180301 00:00:00'
            train_end_date = '20180930 23:00:00'
            test_start_date = '20181001 00:00:00'
            test_end_date = '20181230 23:00:00'
        
        train_x, train_y = X[train_start_date:train_end_date], Y[train_start_date:train_end_date]
        test_x, test_y = X[test_start_date:test_end_date], Y[test_start_date:test_end_date]
    
        #get data without datetime index
        train_x_wo_t = train_x.reset_index().drop(columns=['time'])
        train_y_wo_t = train_y.reset_index().drop(columns=['time'])
        test_x_wo_t = test_x.reset_index().drop(columns=['time'])
        test_y_wo_t = test_y.reset_index().drop(columns=['time'])
    
#%%
        
        for i, estimator in enumerate(estimator_list):
            print("current estimator:", title_list[i])
            res = {}

            # start training model
            start_time = time.time()
            
            model, results, best_param = estimator(train_x_wo_t, train_y_wo_t)
            end_time = time.time()     
            
            if estimator == xgb:
                predict_y_wo_t = model.predict(test_x_wo_t.values)
            else:
                predict_y_wo_t = model.predict(test_x_wo_t)
                
            predict_y = pd.DataFrame(predict_y_wo_t, index=test_y.index)
    
            predict_y = predict_y.rename(columns={0: 'y_' + tag})      
            stitle = title_list[i] + 'Prediction ,station(' + str(sno) +'), '+ tag
            rmse = plot_prediction(stitle,test_y['y_' + tag],train_start_date,train_end_date,predict_y['y_' + tag])
            
            #save result
            res['name'] = stitle
            res['results'] = results
            res['best_param'] = best_param
            res['RMSE'] = rmse
            res['time'] = (end_time-start_time)
            
            print('RMSE:', rmse)   
            print("Spent Time:",end_time-start_time,'sec.')
            
            result_list = result_list + [res]

#%%
print(result_list)