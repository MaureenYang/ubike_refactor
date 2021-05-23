# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 20:56:50 2019

@author: Maureen
"""

from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import PredefinedSplit
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_squared_error
import math

# tools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore')

def index_splitter(N, fold):
    index_split = []
    test_num = int(N/fold)
    train_num = N-test_num

    for i in range(0,train_num):
        index_split.append(-1)

    for i in range(train_num,N):
        index_split.append(0)

    return index_split


def ada(X, Y, kfold=3, feature_set=None):

    arr = index_splitter(N = len(X), fold = kfold)
    ps = PredefinedSplit(arr)

    for train, test in ps.split():
        train_index = train
        test_index = test

    train_X, train_y = X.values[train_index,:], Y.values[train_index]
    test_X, test_y = X.values[test_index,:], Y.values[test_index]
    
    arr = index_splitter(N = len(train_X), fold = kfold)
    ps2 = PredefinedSplit(arr)


    #baseline
    lr_model = LinearRegression()
    ada = AdaBoostRegressor(random_state = 42,loss='square')#,base_estimator=lr_model)
    ada.fit(train_X, train_y)
    
    print('Parameters for baseline:')
    print(ada.get_params())


    #grid search
    lr_log = np.linspace(-8,5,14)

    lr = []
    for i in lr_log:
        a = math.pow(10,i)
        lr = lr + [a]
    
    n_estimators = [int(x) for x in range(20,200,20)] #[int(x) for x in np.linspace(start = 10, stop = 200, num = 50)]
    loss = ['square']
   

    grid_grid = { 'n_estimators': n_estimators,
                    'learning_rate': lr,
                    'loss': loss
                 }


    ada_grid = GridSearchCV(estimator=ada, param_grid=grid_grid, scoring='neg_mean_squared_error', cv = ps2.split(), verbose=2, n_jobs=-1)
    ada_grid.fit(train_X, train_y)
    BestPara_grid = ada_grid.best_params_
    print(ada_grid.best_params_)
    #cv_results_grid = ada_grid.cv_results_
    
    
    
    # second run, grid search
    lr_unit =  BestPara_grid['learning_rate']
    lr = [x for x in np.linspace(start = lr_unit, stop = lr_unit*9, num = 9)]
    
    ets_unit =  BestPara_grid['n_estimators']
    n_estimators = [int(x) for x in range(ets_unit - 20, ets_unit + 20, 5)]

   
    grid_grid2 = {  'learning_rate' : lr,
                    'n_estimators': n_estimators,
                    'loss': loss
                  }
    
    ada_grid2 = GridSearchCV(estimator=ada, param_grid=grid_grid2, scoring='neg_mean_squared_error',
                                  cv = ps2.split(), verbose=2, n_jobs=-1)    
    
    ada_grid2.fit(train_X, train_y)
    print('Ada v2 Parameters:')
    print(ada_grid2.best_params_)
    
    predict_y_base=ada.predict(test_X)
    predict_y_grid=ada_grid.predict(test_X)
    predict_y_grid2 = ada_grid2.predict(test_X)

    errors_Grid_CV = np.sqrt(mean_squared_error(predict_y_grid,test_y))
    errors_Grid2_CV = np.sqrt(mean_squared_error(predict_y_grid2,test_y))
    errors_baseline = np.sqrt(mean_squared_error(predict_y_base,test_y))


    results = [errors_baseline,errors_Grid_CV,errors_Grid2_CV]
    print('Adaboot Results:',results)

    if False:
        #feature importance
        num_feature = len(ada_grid.best_estimator_.feature_importances_)
        plt.figure(figsize=(24,6))
        plt.bar(range(0,num_feature*4,4),ada_grid.best_estimator_.feature_importances_)

        label_name = X.keys()
        plt.xticks(range(0,num_feature*4,4), label_name)
        plt.title("Feature Importances"+",kfold="+str(kfold))
        plt.show()
        

    return ada_grid2.best_estimator_, results, ada_grid2.best_params_
