import pandas as pd
import pickle
import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import PredefinedSplit

# Performance metrics
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_squared_error


def index_splitter(N, fold):
    index_split = []
    test_num = int(N/fold)
    train_num = N-test_num

    for i in range(0,train_num):
        index_split.append(-1)

    for i in range(train_num,N):
        index_split.append(0)

    return index_split


def RMLSE(predict_y_base, test_y):
    errors_baseline = np.sqrt(mean_squared_log_error(predict_y_base,test_y))
    return errors_baseline


source_folder = "csvfile/parsed_station/"
target_pkl_folder = "pkl/"
filesinpath = os.listdir(source_folder)
parsed_file = []
counter = 0

for f in sorted(filesinpath):
    print("file name:",f)
    df = pd.read_csv(source_folder + f)
    df = df.dropna()
    df['time'] = pd.to_datetime(df['time'], format='%Y/%m/%d %H%M%S', errors='ignore')
    df = df.set_index(pd.DatetimeIndex(df['time']))
    X = df.drop(columns = [df.keys()[0],'y_sbi','y_bemp'])
    #print(X.head())

    Y = df['y_bemp']
    Y2 = df['y_sbi']
    sno = str(int(df['sno'][0]))

    # Data Splitter
    #arr = index_splitter(N=len(X), fold=6)
    #ps = PredefinedSplit(arr)

    #for train, test in ps.split():
    #    train_index = train
    #    test_index = test

    #train_X, train_y,train_y2= X.iloc[train_index,:], Y.iloc[train_index], Y2.iloc[train_index]
    #test_X, test_y,test_y2 = X.iloc[test_index,:], Y.iloc[test_index], Y2.iloc[test_index]

    train_X = X["20180301":"20180312"]
    train_y = Y["20180301":"20180312"]
    train_y2 = Y2["20180301":"20180312"]

    test_X = X["20180313":"20180320"]
    test_y = Y["20180313":"20180320"]
    test_y2 = Y2["20180313":"20180320"]

    LR = LinearRegression()
    LR2 = LinearRegression()


    lr = LR.fit(train_X, train_y)
    lr2 = LR2.fit(train_X, train_y2)


    predict_y_base=LR.predict(test_X)
    predict_y2_base=LR2.predict(test_X)

    pickle.dump(LR, open(target_pkl_folder+'bemp_model_'+(sno)+'.pkl','wb'))
    pickle.dump(LR, open(target_pkl_folder+'sbi_model_'+(sno)+'.pkl','wb'))

    errors_baseline = (mean_squared_error(predict_y_base,test_y))#,squared = False))
    errors_baseline2 = (mean_squared_error(predict_y2_base,test_y2))#,squared = False))

    results = [errors_baseline, errors_baseline2]
