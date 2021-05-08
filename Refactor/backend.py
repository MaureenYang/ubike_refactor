import pandas as pd
import numpy as np

import datetime as dt
import json
import time
import os, sys
import shutil
import socket
import data_parser
import pickle


# parameters
csvpath = "csvfile/parsed/final_predict_0301_0305.csv"

try:
    df = pd.read_csv(csvpath,dtype=object)
    df.drop("Unnamed: 0", 1, inplace=True) #merged file need
    df["time"] = pd.to_datetime(df["time"], format="%Y-%m-%d %H:%M:%S")
    df.index = df["time"]
    df.drop("time", 1, inplace=True)
    df.holiday = (df.holiday == 'True').astype(bool)
    df = df.drop_duplicates()
    df = df.sort_index()
except Exception as e:
    print("exception when read csv")
    print(e)


float_list = ['bemp', 'sbi', 'sno', 'HUMD', 'H_24R', 'PRES', 'TEMP', 'WDIR', 'WDSE',
       'PrecpHour', 'UVI', 'Visb', 'WDGust', 'WSGust', 'weekday', 'hours', #'holiday',
       'bemp_1h', 'bemp_2h', 'bemp_3h', 'bemp_4h', 'bemp_5h',
       'bemp_6h', 'bemp_7h', 'bemp_8h', 'bemp_9h', 'bemp_10h', 'bemp_11h',
       'bemp_12h', 'bemp_1d', 'bemp_2d', 'bemp_3d', 'bemp_4d', 'bemp_5d',
       'bemp_6d', 'bemp_7d', 'sbi_1h', 'sbi_2h', 'sbi_3h', 'sbi_4h', 'sbi_5h',
       'sbi_6h', 'sbi_7h', 'sbi_8h', 'sbi_9h', 'sbi_10h', 'sbi_11h', 'sbi_12h',
       'sbi_1d', 'sbi_2d', 'sbi_3d', 'sbi_4d', 'sbi_5d', 'sbi_6d', 'sbi_7d',
       'predict_hour', 'y_bemp', 'y_sbi']

df[float_list] = df[float_list].astype(float)

# can be rewrite by use Youbike processor
# json_info_list: list of dictionary data of each station
# text_list : data shows in hover box
def getStationInformation():  # can be rewrite by use Youbike processor

    json_file = open("raw_data/YouBikeTPNow.json", encoding='utf-8')
    ubike_data = json.load(json_file)
    json_info_list = []
    i = 0
    for key,value in ubike_data['retVal'].items():
        i = i + 1
        sno = value['sno']
        sna = value['sna']
        tot = value['tot']
        sbi = value['sbi']
        sarea = value['sarea']
        mday = value['mday']
        lat = value['lat']
        lng = value['lng']
        ar = value['ar']
        sareaen = value['sareaen']
        snaen = value['snaen']
        aren = value['aren']
        bemp = value['bemp']
        act = value['act']
        value['idx'] = i
        json_info_list = json_info_list + [value]

    return json_info_list#, text_list


model = model2 = pickle.load(open('D:/youbike/code/ubike_refactor/Refactor/pkl/GB_stat0.pkl','rb'))
#model2 = pickle.load(open('sbi_model.pkl','rb'))

X = df.drop(columns = ['y_bemp','y_sbi'])
X = X.drop(columns=['bemp_1h', 'bemp_2h', 'bemp_3h', 'bemp_4h', 'bemp_5h','bemp_6h', 'bemp_7h', 'bemp_8h', 'bemp_9h', 'bemp_10h', 'bemp_11h','bemp_12h', 'bemp_1d', 'bemp_2d', 'bemp_3d', 'bemp_4d', 'bemp_5d','bemp_6d', 'bemp_7d','bemp'])

Y= df['y_bemp']
Y2= df['y_sbi']

predict_y = model.predict(X)
predict_y2 = model2.predict(X)
print('predict_y:',predict_y)

from sklearn.metrics import mean_squared_error
errors_baseline = (mean_squared_error(predict_y,Y))
errors_baseline2 = (mean_squared_error(predict_y2,Y2))
results = [errors_baseline, errors_baseline2]

df['y_bemp_predict'] = pd.Series(predict_y,index = df.index).astype(int)
df['y_sbi_predict'] = pd.Series(predict_y2,index = df.index).astype(int)

'''
    input value: sno, datePicked
    output value:
        - the +/- 12 hour value for the date datePicked
        - the predicted 12 hour result from datePicked
'''

def getBikeDatabyDate(sno,datePicked):
    start_time = end_time = datePicked
    try:
        #print("get Bike data by Date:",datePicked)
        selected_df = df[df['sno'] == float(sno)]
        time_interval = dt.timedelta(hours=12)
        start_time = start_time - time_interval
        end_time = end_time + time_interval
        #print("start time:", start_time)
        #print("end time:", end_time)
        #print("datePicked:", datePicked)

        start_dt_idx = start_time.strftime("%Y-%m-%d %H:%M:%S")
        end_dt_idx = end_time.strftime("%Y-%m-%d %H:%M:%S")
        piked_dt_idx = datePicked.strftime("%Y-%m-%d %H:%M:%S")

        empty_df = selected_df[start_dt_idx:end_dt_idx].bemp.astype(int).drop_duplicates()
        bike_df = selected_df[start_dt_idx:end_dt_idx].sbi.astype(int).drop_duplicates()

        true_bemp_df = selected_df[selected_df.index == piked_dt_idx].astype(int)
        true_sbi_df = selected_df[selected_df.index == piked_dt_idx].astype(int)

        true_bemp_df = true_bemp_df[['predict_hour','y_bemp']]
        true_sbi_df = true_sbi_df[['predict_hour','y_sbi']]

        pre_empty_df = selected_df[selected_df.index == piked_dt_idx][['predict_hour','y_bemp_predict']]
        pre_bike_df = selected_df[selected_df.index == piked_dt_idx][['predict_hour','y_sbi_predict']]

        return [empty_df,bike_df,pre_empty_df,pre_bike_df,true_bemp_df,true_sbi_df]

    except Exception as e:
        print("ERROR in getBikeDatabyDate")
        print(e)

    return None

#get current weather by station ID

def get_weather_by_station(sno):

    par = data_parser.data_parser()
    par.update_current_data()
    bw_df = par.generate_bike_weather_data()
    bw_df_sno = bw_df[bw_df['sno'] == sno]

    bw_df_sno = bw_df_sno[['WDIR', 'WDSE', 'TEMP',
       'HUMD', 'PRES', 'H_24R', 'WSGust', 'WDGust', 'UVI', 'PrecpHour','Visb']]

    return bw_df_sno


if __name__ == '__main__':
    #print(df.columns)
    selected_df = df[df['sno'] == 1]
    #print(selected_df['20180302'][['predict_hour','y_bemp','y_bemp_predict','bemp']])
    # bemp: the truth for now: the datePicked -> for drawing
    #y_bemp: the truth for predict hour: datePicked + predict hour (Truth) : for showing the truth
    #y_bemp_predict: the result we predict:(prediction): for drawing and show the prediction
    a = getBikeDatabyDate(1,dt.datetime.strptime("2018-03-02 12:00:00", "%Y-%m-%d %H:%M:%S"))
    #print(a[2]['2018-03-02 01:00:00'])
    #get_weather_by_station(1)