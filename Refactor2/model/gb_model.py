# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 20:56:50 2019

@author: Mandy

"""

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import PredefinedSplit
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_squared_error

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math


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
def gb(X, Y, kfold=3, feature_set=None):
    arr = index_splitter(N = len(X), fold = kfold)
    ps = PredefinedSplit(arr)

    for train, test in ps.split():
        train_index = train
        test_index = test


    train_X, train_y = X.values[train_index,:], Y.values[train_index]
    test_X, test_y = X.values[test_index,:], Y.values[test_index]
    arr = index_splitter(N = len(train_X), fold = kfold)
    ps2 = PredefinedSplit(arr)

    # grid search
    lr_log = np.linspace(-8,5,14)

    lr = []
    for i in lr_log:
        a = math.pow(10,i)
        lr = lr + [a]
        
    n_estimators = [int(x) for x in np.linspace(start = 10, stop = 200, num = 50)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [3, 5, 10, 20, 50]
   # max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True]

    # Create the random grid
    random_grid = {'learning_rate' : lr,
                   'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   }

    gb = GradientBoostingRegressor(random_state = 42)

    from pprint import pprint

    # Look at parameters used by our current forest
    print('Parameters currently in use:\n')
    pprint(gb.get_params())
    # Fit the base line search model
    gb.fit(train_X, train_y)
    
    
    # Use the random grid to search for best hyperparameters
    # First create the base model to tune

    # Random search of parameters, using 3 fold cross validation,
    # search across 100 different combinations, and use all available cores
    '''
    gb_random = RandomizedSearchCV(estimator=gb, n_iter=200, param_distributions=random_grid, scoring='neg_mean_squared_error',
                                  cv = ps2.split(), verbose=2, random_state=42, n_jobs=-1)

    # Fit the random search model
    gb_random.fit(train_X, train_y)
    pprint(gb_random.best_params_)
    cv_result_rd= gb_random.cv_results_

    BestPara_random = gb_random.best_params_

    ## Grid search of parameters, using 3 fold cross validation based on Random search
    from sklearn.model_selection import GridSearchCV

    # Number of trees in random forest
    lr = [BestPara_random['learning_rate']]
    n_estimators = [int(x) for x in range(BestPara_random["n_estimators"]-100,BestPara_random["n_estimators"]+100,50)]
    # Number of features to consider at every split
    max_features = [BestPara_random["max_features"]]
    # Maximum number of levels in tree
    #max_depth = [int(x) for x in range(BestPara_random["max_depth"]-10,BestPara_random["max_depth"]+10,2)]
    max_depth = []
    for x in range(BestPara_random["max_depth"]-5,BestPara_random["max_depth"]+5,5):
        if x>0:
            max_depth.append(int(x))
    # Minimum number of samples required to split a node
    min_samples_split = []
    for x in range(BestPara_random["min_samples_split"]-2,BestPara_random["min_samples_split"]+2,2):
        if x>1:
            min_samples_split.append(int(x))
    # Minimum number of samples required at each leaf node
    min_samples_leaf = []
    for x in range(BestPara_random["min_samples_leaf"]-1,BestPara_random["min_samples_leaf"]+1,1):
        if x>0:
            min_samples_leaf.append(int(x))

    '''
    # Create the random grid
    grid_grid = { 'learning_rate' : lr,
                  'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                  }
    
    gb_grid = GridSearchCV(estimator=gb, param_grid=grid_grid, scoring='neg_mean_squared_error',
                                  cv = ps2.split(), verbose=2, n_jobs=-1)
    # Fit the grid search model
    gb_grid.fit(train_X, train_y)
    BestPara_grid = gb_grid.best_params_

    pprint(gb_grid.best_params_)
    cv_results_grid = gb_grid.cv_results_



    #prediction
    #predict_y=gb_random.predict(test_X)
    predict_y_grid=gb_grid.predict(test_X)
    predict_y_base=gb.predict(test_X)
    # Performance metrics
    from sklearn.metrics import mean_squared_log_error
    from sklearn.metrics import mean_squared_error

    def RMLSE(predict_y_grid, predict_y, predict_y_base, test_y):
        errors_Grid_CV = np.sqrt(mean_squared_log_error(predict_y_grid,test_y))
        #errors_Random_CV = np.sqrt(mean_squared_log_error(predict_y,test_y))
        errors_baseline = np.sqrt(mean_squared_log_error(predict_y_base,test_y))
        return errors_Grid_CV, errors_baseline

    errors_Grid_CV = (mean_squared_error(predict_y_grid,test_y))##,squared = False))
    #errors_Random_CV = (mean_squared_error(predict_y,test_y))##,squared = False))
    errors_baseline = (mean_squared_error(predict_y_base,test_y))##,squared = False))

    x_axis = range(2)
    results = [errors_Grid_CV,errors_baseline]
    print('gradient boost results:',results)

    if False:
        fig=plt.figure(figsize=(15,8))
        x_axis = range(3)
        plt.bar(x_axis, results)
        plt.xticks(x_axis, ('GridSearchCV','RandomizedSearchCV', 'Baseline'))
        #plt.show()
        plt.savefig('gb_compare_error.png')

        #feature importance
        num_feature = len(gb_grid.best_estimator_.feature_importances_)
        plt.figure(figsize=(24,6))
        plt.bar(range(0,num_feature*4,4),gb_grid.best_estimator_.feature_importances_)

        label_name = X.keys()

        plt.xticks(range(0,num_feature*4,4), label_name)
        plt.title("Feature Importances"+",kfold="+str(kfold))
        #plt.show()
        plt.savefig('gb_feature_importance.png')

        fig=plt.figure(figsize=(20,8))
        ax = fig.gca()
        x_label = range(0,len(predict_y_grid))
        plt.title("kfold="+str(kfold))
        ax.plot(x_label, predict_y_grid, 'r--', label = "predict")
        ax.plot(x_label, test_y, label = "ground_truth")
        ax.set_ylim(0, 200)
        ax.legend()
        #plt.show()
        plt.savefig('gb_prediction.png')

    return gb_grid.best_estimator_
