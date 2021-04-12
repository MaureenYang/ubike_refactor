# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 16:44:02 2020

@author: Mandy
"""
#import Models.lr_model as lr
import Models.lasso_model as lasso
import Models.ridge_model as ridge
import Models.rf_model as rf
import Models.gb_model as gb
import Models.ada_model as ada


from sklearn.metrics import mean_squared_error
from sklearn.model_selection import PredefinedSplit

import matplotlib.pyplot as plt
import pandas as pd

def model_score(X_test, Y_test, model):
    Y_predict = model(X_test)
    RMSE = (mean_squared_error(Y_predict, Y_test))
    return RMSE

def index_splitter(N, fold):
    index_split = []
    test_num = int(N/fold)
    train_num = N-test_num

    for i in range(0,train_num):
        index_split.append(-1)

    for i in range(train_num,N):
        index_split.append(0)

    return index_split


####################
## Decide your Data Set

xaxis_Name = []
model_score_list = []

if False:
    estimator_name_list = ["LinearReg","Ridge", "Lasso", "Gradient Boost", "RandomForest", "AdaBoost"]
    estimator_list = [lr.lr,ridge.ridge, lasso.lasso, gb.gb, rf.rf, ada.ada]
    line_style = ['-xb','-xr','-xg','-xc', '-xm','-xy']
else:
    estimator_name_list = ["Ridge", "Lasso", "Gradient Boost", "RandomForest", "AdaBoost"]
    estimator_list = [ridge.ridge, lasso.lasso, gb.gb, rf.rf, ada.ada]
    line_style = ['-xr','-xg','-xc', '-xm','-xy']

target = 'bemp_'
source_folder = "csvfile/parsed/"
file_name = "new_data_sno1"
file_name_list = []


for i in range(6):
    target_label = str(i+1) + "h"
    #file_name_list.append(source_folder+"\\"+file_name+"_predict_"+target_label+".csv")
    filename = source_folder+file_name+"_predict_"+target+target_label+".csv"
    file_name_list.append(filename)
    xaxis_Name.append(target_label)


df_list = []
for feature_name in file_name_list:
    df_list.append(pd.read_csv(feature_name))
    N = len(df_list[(len(df_list)-1)])

df = pd.read_csv(source_folder+"new_data_sno244_predict.csv")

df = df.dropna()
X = df.drop(columns = [df.keys()[0],'sbi','bemp','time'])
Y = df['bemp']


# Data Splitter
arr = index_splitter(N=len(X), fold=6)
ps = PredefinedSplit(arr)

for train, test in ps.split():
    train_index = train
    test_index = test

#print(train_index,test_index)
train_X, train_y = X.iloc[train_index,:], Y.iloc[train_index]
test_X, test_y = X.iloc[test_index,:], Y.iloc[test_index]
model_score_matrix=[]


##Train Model
for i, estimator in enumerate(estimator_list):

    print("Current Model:",estimator_name_list[i])
    model_predict, model = estimator(train_X, train_y)

    ##Score
    model_score_list = []

    for df_tmp in df_list:    # Data Splitter
        df_tmp = df_tmp.dropna()
        X = df_tmp.drop(columns=[df_tmp.keys()[0],'time.1', 'sbi', 'bemp'])
        Y = df_tmp['bemp']
        arr = index_splitter(N=len(X), fold=6)
        ps = PredefinedSplit(arr)

        for train, test in ps.split():
            train_index = train
            test_index = test

        # test data
        train_X_tmp, train_y_tmp = X.iloc[train_index,:], Y.iloc[train_index]
        test_X_tmp, test_y_tmp = X.iloc[test_index,:], Y.iloc[test_index]

        # Performance Metric
        model_score_list.append(model_score(test_X_tmp, test_y_tmp, model_predict))

    model_score_matrix.append(model_score_list)

fig=plt.figure(figsize=(100,8))
x_axis = [i for i in range(len(file_name_list))]

for i, y in enumerate(model_score_matrix):
    plt.plot(x_axis, y, line_style[i], label=estimator_name_list[i])

plt.xticks(x_axis,  xaxis_Name)
plt.ylabel("RMSE")
plt.legend()
#plt.show()
plt.savefig('model_compare.png')
