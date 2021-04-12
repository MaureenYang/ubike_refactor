import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.model_selection import PredefinedSplit
from sklearn.linear_model import LinearRegression
import os

try:
    srcfilepath = "csvfile/parsed/"
    filesinpath = os.listdir(srcfilepath)
    parsed_file = []
    i = 0
    df = pd.read_csv(srcfilepath + "final_predict.csv")


    def index_splitter(N, fold):
        index_split = []
        test_num = int(N/fold)
        train_num = N-test_num

        for i in range(0,train_num):
            index_split.append(-1)

        for i in range(train_num,N):
            index_split.append(0)

        return index_split

    # preprocessing
    df = df.dropna()
    df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S', errors='ignore')
    df = df.set_index(pd.DatetimeIndex(df['time']))
    X = df.drop(columns = [df.keys()[0],'sbi','bemp','time', 'sbi_1h', 'sbi_2h', 'sbi_3h', 'sbi_4h', 'sbi_5h', 'sbi_6h',
       'sbi_7h', 'sbi_8h', 'sbi_9h', 'sbi_10h', 'sbi_11h', 'sbi_12h', 'sbi_1d',
       'sbi_2d', 'sbi_3d', 'sbi_4d', 'sbi_5d', 'sbi_6d', 'sbi_7d', 'y_sbi'])
    Y = df['bemp']
    print(X.columns)

    # Data Splitter
    arr = index_splitter(N=len(X), fold=3)
    ps = PredefinedSplit(arr)

    for train, test in ps.split():
        train_index = train
        test_index = test

    train_X, train_y = X.iloc[train_index,:], Y.iloc[train_index]
    test_X, test_y = X.iloc[test_index,:], Y.iloc[test_index]

    if True:
        regressor = LinearRegression()
        regressor.fit(train_X, train_y)
        pickle.dump(regressor, open('bemp_model.pkl','wb'))

    model = pickle.load(open('bemp_model.pkl','rb'))

    print(model.predict(test_X))

except Exception as e:
    print(e)
