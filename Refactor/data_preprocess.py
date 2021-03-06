import numpy as np                               # vectors and matrices
import pandas as pd                              # tables and data manipulations
import matplotlib.pyplot as plt                  # plots
import math
import os

srcfilepath = "E:/csvfile/source/"
filesinpath = os.listdir(srcfilepath)
parsed_file = []
counter = 0
final_df = pd.DataFrame()
#for i in range(1,388):
#    new_name = "data_sno_"+str(i).zfill(3)+".csv"
#    print(new_name)
#    parsed_file = parsed_file + [new_name]

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
    #t_idx = (uvi_c.index.hour.values > 6) & (uvi_c.index.hour.values < 19)
    #uvi_c[t_idx].replace(to_replace=0, method='ffill')
    df.UVI = uvi_c
    return df

for f in sorted(filesinpath):
    try:
        counter = counter + 1
        if f in parsed_file:
            print("parsed already:",f)
            continue
        print(f)
        namelist = f.split(".")
        sbi_newfilename = "new_" + namelist[0]+"_sbi_predict.csv"
        bemp_newfilename = "new_" + namelist[0]+"_bemp_predict.csv"

        str_file = srcfilepath + f
        df = pd.read_csv(str_file)
        df = df.rename(columns={"Unnamed: 0": "time"})

        emptyidx=[]
        for x in df.index:
            if df.time[x] is np.nan:
                emptyidx = emptyidx + [x]

        df = df.drop(index = emptyidx)
        df = df.drop(columns = ['act','lat','lng','tot','SeaPres','GloblRad','CloudA','td'])

        df['time'] = pd.to_datetime(df['time'], format='%Y/%m/%d %H%M%S', errors='ignore')
        df = df.set_index(pd.DatetimeIndex(df['time']))
        df = df.drop_duplicates()
        float_idx = ['sno','HUMD','bemp','sbi','PRES', 'TEMP', 'WDIR', 'H_24R', 'WDSE', 'WSGust', 'PrecpHour', 'UVI', 'Visb', 'WDGust']

        df[float_idx] = df[float_idx].astype('float')

        df['bemp'] = df['bemp'].apply(lambda x: round(x, 0))
        df['sbi'] = df['sbi'].apply(lambda x: round(x, 0))

        fill_past_mean_tag = ['bemp','sbi']
        interpolate_tag = ['TEMP','WDIR','H_24R','sno','PRES','HUMD','WDSE'] #CloudA
        fillzero_tag = ['UVI'] #GloblRad

        for tag in fill_past_mean_tag:
            dfl = []
            ndf = df[tag]
            for month in range(0,11):
                x = month%3
                y = math.floor(month/3)
                data = ndf[ndf.index.month == (month+2)]
                idx = data.index[data.apply(np.isnan)]

                #get mean of each weekday
                meanss = []
                for wkday in range(0,7):
                      for hr in range(0,24):
                        means = round(data[(data.index.hour == hr)&(data.index.weekday == wkday)].mean())
                        meanss = meanss + [means]

                #replace na data
                for i in idx:
                    data.loc[i] = meanss[i.weekday()*23 + i.hour]

                dfl = dfl + [data]

            new_df = pd.concat(dfl)
            df[tag]= new_df.values


        for tag in interpolate_tag:
            df[tag] = df[tag].interpolate()


        for tag in fillzero_tag:
            df[tag] = df[tag].fillna(0)

        df['weekday'] = df.index.weekday
        df['hours'] = df.index.hour

        #can be replace by list of anything
        from datetime import date
        from workalendar.asia import Taiwan
        cal = Taiwan()
        holidayidx = []
        for t in cal.holidays(2018):
            date_str = t[0].strftime("%Y-%m-%d")
            holidayidx = holidayidx + [date_str]

        df['holiday'] = df.index.isin(holidayidx)

        df = update_uvi_category(df)
        t_idx = (df.UVI.index.hour.values > 6) & (df.UVI.index.hour.values < 19)
        uvi = df.UVI[t_idx].replace(to_replace=0, method='bfill')
        print(uvi.head(24))
        df.UVI[t_idx] = uvi

        #add historical value
        for tag in ['bemp','sbi']:

            for i in range(1,13):
                df[tag+'_'+str(i)+'h'] = df[tag].shift(i)

            for i in range(1,8):
                df[tag+'_'+str(i)+'d'] = df[tag].shift(i*24)

        #add predict value
        ndf = df
        ndf['predict_hour'] = 1
        for tag in ['bemp','sbi']:
            ndf['y_' + tag] = df[tag].shift(-1)

        for i in range(1,13):
            ndf2 = df
            ndf2['predict_hour'] = i
            for tag in ['bemp','sbi']:
                ndf2['y_' + tag] = df[tag].shift(-i)
                #ndf2.to_csv("csvfile/parsed/" + "new_" + newfilename +"_predict_"+tag+"_"+str(i)+"h.csv")
            ndf = ndf.append(ndf2, ignore_index=True)

        ndf = ndf.dropna()

        ndf["time"] = pd.to_datetime(ndf["time"], format="%Y-%m-%d %H:%M")
        ndf.index = ndf["time"]

        #ndf = ndf["20180301":"20181231"]

        bemp_ndf = ndf[['time', 'bemp', 'sno', 'HUMD', 'H_24R', 'PRES', 'TEMP', 'WDIR',
       'WDSE', 'PrecpHour', 'UVI', 'Visb', 'WDGust', 'WSGust', 'weekday',
       'hours', 'holiday', 'bemp_1h', 'bemp_2h', 'bemp_3h', 'bemp_4h',
       'bemp_5h', 'bemp_6h', 'bemp_7h', 'bemp_8h', 'bemp_9h', 'bemp_10h',
       'bemp_11h', 'bemp_12h', 'bemp_1d', 'bemp_2d', 'bemp_3d', 'bemp_4d',
       'bemp_5d', 'bemp_6d', 'bemp_7d','predict_hour', 'y_bemp']]

        sbi_ndf = ndf[['time', 'sbi', 'sno', 'HUMD', 'H_24R', 'PRES', 'TEMP', 'WDIR',
       'WDSE', 'PrecpHour', 'UVI', 'Visb', 'WDGust', 'WSGust', 'weekday',
       'hours', 'holiday', 'sbi_1h', 'sbi_2h', 'sbi_3h', 'sbi_4h',
       'sbi_5h', 'sbi_6h', 'sbi_7h', 'sbi_8h', 'sbi_9h', 'sbi_10h', 'sbi_11h',
       'sbi_12h', 'sbi_1d', 'sbi_2d', 'sbi_3d', 'sbi_4d', 'sbi_5d', 'sbi_6d',
       'sbi_7d', 'predict_hour', 'y_sbi']]

        #just one file
        target_path = "E:/csvfile/parsed2/"
        #final_df  = final_df.append(ndf,ignore_index=True)
        bemp_ndf.to_csv(target_path +"bemp/"+bemp_newfilename)
        sbi_ndf.to_csv(target_path +"sbi/"+sbi_newfilename)
        #final_df  = final_df.append(ndf,ignore_index=True)


    except Exception as e:
        print('ERROR:',e)
    break

#final_df.to_csv("csvfile/parsed/final_predict_0301_0305.csv")
