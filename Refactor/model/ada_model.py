# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 20:56:50 2019

@author: Mandy
"""

from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import PredefinedSplit
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_squared_error

# tools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pprint import pprint
graph = False

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

    learning_rate = [x for x in np.linspace(0.1, 1, num = 10)]
    n_estimators = [int(x) for x in np.linspace(start = 20, stop = 1000, num = 100)]
    loss = ['square']

    random_grid = { 'n_estimators': n_estimators,
                    'learning_rate': learning_rate,
                    'loss': loss}

    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    ada = AdaBoostRegressor(random_state = 42,loss='square')

    # Look at parameters used by our current forest
    print('Parameters for baseline:\n')
    pprint(ada.get_params())

    # Random search of parameters, using 3 fold cross validation,
    # search across 100 different combinations, and use all available cores
    ada_random = RandomizedSearchCV(estimator=ada, n_iter=200, param_distributions = random_grid, scoring='neg_mean_squared_error',
                                  cv = ps2.split(), verbose=2, random_state=42, n_jobs=-1)

    # Fit the random search model
    ada_random.fit(train_X, train_y)
    pprint(ada_random.best_params_)

    cv_result_rd= ada_random.cv_results_
    BestPara_random = ada_random.best_params_


    ## Grid search of parameters, using 3 fold cross validation based on Random search
    lr = [BestPara_random['learning_rate']]
    #n_estimators = [BestPara_random["n_estimators"]]

    n_estimators = [int(x) for x in range(BestPara_random["n_estimators"]-10, BestPara_random["n_estimators"]+10, 20)]
    n_estimators = [item for item in n_estimators if item > 0 ]

    grid_grid = { 'n_estimators': n_estimators,
                  'learning_rate': lr,
                  'loss': loss}

    ada_grid = GridSearchCV(estimator=ada, param_grid=grid_grid, scoring='neg_mean_squared_error',
                                  cv = ps2.split(), verbose=2, n_jobs=-1)

    # Fit the grid search model
    ada_grid.fit(train_X, train_y)
    BestPara_grid = ada_grid.best_params_

    pprint(ada_grid.best_params_)
    cv_results_grid = ada_grid.cv_results_

    # Fit the base line search model
    ada.fit(train_X, train_y)

    #prediction
    predict_y=ada_random.predict(test_X)
    predict_y_grid=ada_grid.predict(test_X)
    predict_y_base=ada.predict(test_X)

    # Performance metrics

    def RMLSE(predict_y_grid, predict_y, predict_y_base, test_y):
        errors_Grid_CV = np.sqrt(mean_squared_log_error(predict_y_grid,test_y))
        errors_Random_CV = np.sqrt(mean_squared_log_error(predict_y,test_y))
        errors_baseline = np.sqrt(mean_squared_log_error(predict_y_base,test_y))
        return errors_Grid_CV, errors_Random_CV, errors_baseline

    errors_Grid_CV = (mean_squared_error(predict_y_grid,test_y))#,squared = False))
    errors_Random_CV = (mean_squared_error(predict_y,test_y))#,squared = False))
    errors_baseline = (mean_squared_error(predict_y_base,test_y))#,squared = False))

    x_axis = range(3)
    results = [errors_Grid_CV,errors_Random_CV,errors_baseline]

    print('Adaboot Results:',results)

    if True:
        fig=plt.figure(figsize=(15,8))
        x_axis = range(3)
        plt.bar(x_axis, results)
        plt.xticks(x_axis, ('GridSearchCV','RandomizedSearchCV', 'Baseline'))
        #plt.show()
        plt.savefig('ada_compare_error.png')

        #feature importance
        num_feature = len(ada_grid.best_estimator_.feature_importances_)
        plt.figure(figsize=(24,6))
        plt.bar(range(0,num_feature*4,4),ada_grid.best_estimator_.feature_importances_)

        label_name = X.keys()

        plt.xticks(range(0,num_feature*4,4), label_name)
        plt.title("Feature Importances"+",kfold="+str(kfold))
        #plt.show()
        plt.savefig('ada_feature_importance.png')

        fig=plt.figure(figsize=(20,8))
        ax = fig.gca()
        x_label = range(0,len(predict_y_grid))
        plt.title("kfold="+str(kfold))
        ax.plot(x_label, predict_y_grid, 'r--', label = "predict")
        ax.plot(x_label, test_y, label = "ground_truth")
        ax.set_ylim(0, 200)
        ax.legend()
        #plt.show()
        plt.savefig('ada_prediction.png')

        #return a dictionary for all results
    return ada_grid.predict,ada_grid.best_estimator_
