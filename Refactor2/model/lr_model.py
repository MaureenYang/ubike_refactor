# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 20:56:50 2019

@author: Mandy
"""

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import PredefinedSplit

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pprint import pprint

def index_splitter(N, fold):
    index_split = []
    test_num = int(N/fold)
    train_num = N-test_num

    for i in range(0,train_num):
        index_split.append(-1)

    for i in range(train_num,N):
        index_split.append(0)

    return index_split


# Number of trees in random forest
def lr(X, Y, kfold=3, feature_set=None):
    arr = index_splitter(N = len(X), fold = kfold)
    ps = PredefinedSplit(arr)

    for train, test in ps.split():
        train_index = train
        test_index = test


    train_X, train_y = X.values[train_index,:], Y.values[train_index]
    test_X, test_y = X.values[test_index,:], Y.values[test_index]
    arr = index_splitter(N = len(train_X), fold = kfold)
    ps2 = PredefinedSplit(arr)


    LR = LinearRegression()


    print('Parameters currently in use:\n')
    pprint(LR.get_params())

    lr = LR.fit(train_X, train_y)
    pprint(lr.coef_)

    BestPara_random = lr.coef_
    print(BestPara_random)

    predict_y_base=LR.predict(test_X)

    # Performance metrics
    from sklearn.metrics import mean_squared_log_error
    from sklearn.metrics import mean_squared_error

    def RMLSE(predict_y_base, test_y):
        errors_baseline = np.sqrt(mean_squared_log_error(predict_y_base,test_y))
        return errors_baseline

    errors_baseline = (mean_squared_error(predict_y_base,test_y))#,squared = False))
    results = [errors_baseline]
    print('Linear Reg results:',results)

    if True:
        fig=plt.figure(figsize=(20,8))
        x_axis = range(1)

        #feature importance
        #num_feature = len(LR.best_estimator_.feature_importances_)
        #plt.figure(figsize=(24,6))
        #plt.bar(range(0,num_feature*4,4),LR.best_estimator_.feature_importances_)

        #label_name = X.keys()
        #plt.xticks(range(0,num_feature*4,4), label_name)
        #plt.savefig('LR_feature_importance.png')

        fig=plt.figure(figsize=(20,8))
        ax = fig.gca()
        x_label = range(0,len(predict_y_base))
        plt.title("kfold="+str(kfold))
        ax.plot(x_label, predict_y_base, 'r--', label = "predict")
        ax.plot(x_label, test_y, label = "ground_truth")
        ax.set_ylim(0, 200)
        ax.legend()
        #plt.show()


    return lr.predict,lr.coef_
