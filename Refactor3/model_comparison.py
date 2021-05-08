import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pymongo
import os,sys
import time
from sklearn.metrics import mean_squared_error
import data_preprocessor as dp
sys.path.append("model/")
from ridge_model import ridge
from lasso_model import lasso
import baseline

#estimator_list = [baseline.HistroicalAverage,baseline.MovingAverage,baseline.SARIMA, lasso, ridge]
#estimator_list = [baseline.SARIMA]
estimator_list = [baseline.HistroicalAverage,baseline.MovingAverage, lasso, ridge]
predict_bemp_flag = True
predict_sbi_flag = False


def update_uvi_category(df):
    UVI_val_catgory = {'>30': 8, '21-30':7, '16-20':6, '11-15':5,'7-10':4, '3-6':3, '1-2':2,'<1':1,'0':0}
    UVI_catgory = {'uvi_30': 8, 'uvi_20_30': 7, 'uvi_16_20': 6, 'uvi_11_16': 5, 'uvi_7_11': 4, 'uvi_3_7': 3, 'uvi_1_3': 2, 'uvi_1': 1, 'uvi_0': 0}
    uvi_c = pd.Series(index = df.index,data=[None for _ in range(len(df.index))])
    uvi_c[df.UVI > 30] = 'uvi_30'
    idx = (df.UVI > 20) & (df.UVI <= 30)
    uvi_c[idx] = 'uvi_20_30'
    idx = (df.UVI > 16) & (df.UVI <= 20)
    uvi_c[idx] = 'uvi_16_20'
    idx = (df.UVI > 11) & (df.UVI <= 16)
    uvi_c[idx] = 'uvi_11_16'
    idx = (df.UVI > 7) & (df.UVI <= 11)
    uvi_c[idx] = 'uvi_7_11'
    idx = (df.UVI > 3) & (df.UVI <= 7)
    uvi_c[idx] = 'uvi_3_7'
    idx = (df.UVI > 1) & (df.UVI <= 3)
    uvi_c[idx] = 'uvi_1_3'
    uvi_c[df.UVI <= 1] = 'uvi_1'
    uvi_c[df.UVI == 0] = 'uvi_0'

    uvi_c = uvi_c.map(UVI_catgory)
    df.UVI = uvi_c
    return df

if __name__ == "__main__":
    # read data from file
    mse_list = []
    for i, estimator in enumerate(estimator_list):
        if predict_bemp_flag:
            filepath = "E:/csvfile/parsed/bemp/"
            filesinpath = os.listdir(filepath)
            for f in sorted(filesinpath):           #for each file, ran model
                print("file name:", f)
                df = pd.read_csv(filepath + f)
                df['time'] = pd.to_datetime(df['time'], format='%Y/%m/%d %H%M%S', errors='ignore')
                df = df.set_index(pd.DatetimeIndex(df['time'])).sort_index()
                new_df = update_uvi_category(df)

                X = new_df.drop(columns=['time','time.1','y_bemp'])
                Y = new_df.y_bemp

                # Data Splitter
                start_date = '20180301'
                split_date = '20180930'
                train_x, train_y = X[start_date:split_date], Y[start_date:split_date]
                test_x, test_y = X[split_date:], Y[split_date:]

                train_x = train_x.reset_index().drop(columns=['time'])
                train_y = train_y.reset_index().drop(columns=['time'])
                test_x = test_x.reset_index().drop(columns=['time'])
                test_y = test_y.reset_index().drop(columns=['time'])
                start_time = time.time()
                if estimator not in [baseline.HistroicalAverage, baseline.MovingAverage, baseline.SARIMA]:
                    model = estimator(train_x,train_y)
                    predict_y = model.predict(test_x)
                    mse = np.sqrt(mean_squared_error(test_y,predict_y))
                else:
                    mse = estimator(train_y,test_y)
                end_time = time.time()
                print("Spent Time:",end_time-start_time,'sec.')
                mse_list = mse_list + [mse]
                break


        if predict_sbi_flag:
            filepath = "E:/csvfile/parsed/sbi/"
            filesinpath = os.listdir(filepath)
            for f in sorted(filesinpath):
                pass

    print(mse_list)
