# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 20:56:50 2019

@author: Mandy
"""
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import PredefinedSplit

from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_squared_error

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
def ridge(X, Y, kfold=3, feature_set=None):
    arr = index_splitter(N = len(X), fold = kfold)
    ps = PredefinedSplit(arr)

    for train, test in ps.split():
        train_index = train
        test_index = test

    train_X, train_y = X.values[train_index,:], Y.values[train_index]
    test_X, test_y = X.values[test_index,:], Y.values[test_index]
    arr = index_splitter(N = len(train_X), fold = kfold)
    ps2 = PredefinedSplit(arr)

    # Create the random grid
    alpha = np.linspace(1,100,100)

    random_grid = {'alpha': alpha}

    ridge = Ridge(random_state = 42)

    # Look at parameters used by our current forest
    print('Parameters currently in use:\n')
    pprint(ridge.get_params())

    # Use the random grid to search for best hyperparameters
    # First create the base model to tune

    # Random search of parameters, using 3 fold cross validation,
    # search across 100 different combinations, and use all available cores
    ridge_random = RandomizedSearchCV(estimator=ridge, param_distributions=random_grid, scoring='neg_mean_squared_error',
                                  cv = ps2.split(), verbose=2, random_state=42, n_jobs=-1)

    # Fit the random search model
    ridge_random.fit(train_X, train_y)
    pprint(ridge_random.best_params_)
    cv_result_rd= ridge_random.cv_results_

    BestPara_random = ridge_random.best_params_
    print(BestPara_random)

    ## Grid search of parameters, using 3 fold cross validation based on Random search
    from sklearn.model_selection import GridSearchCV

    # Number of trees in random forest
    alpha = np.linspace(BestPara_random["alpha"]-1,BestPara_random["alpha"]+1,10)

    # Create the random grid
    grid_grid = {'alpha': alpha}

    ridge_grid = GridSearchCV(estimator=ridge, param_grid=grid_grid, scoring='neg_mean_squared_error',
                                  cv = ps2.split(), verbose=2, n_jobs=-1)
    # Fit the grid search model
    ridge_grid.fit(train_X, train_y)
    BestPara_grid = ridge_grid.best_params_

    pprint(ridge_grid.best_params_)
    cv_results_grid = ridge_grid.cv_results_

    # Fit the base line search model
    ridge.fit(train_X, train_y)

    #prediction
    predict_y=ridge_random.predict(test_X)
    predict_y_grid=ridge_grid.predict(test_X)
    predict_y_base=ridge.predict(test_X)
    # Performance metrics

    def RMLSE(predict_y_grid, predict_y, predict_y_base, test_y):
        errors_Grid_CV = np.sqrt(mean_squared_log_error(predict_y_grid,test_y))
        errors_Random_CV = np.sqrt(mean_squared_log_error(predict_y,test_y))
        errors_baseline = np.sqrt(mean_squared_log_error(predict_y_base,test_y))
        return errors_Grid_CV, errors_Random_CV, errors_baseline


    errors_Grid_CV = (mean_squared_error(predict_y_grid,test_y))#,squared = False))
    errors_Random_CV = (mean_squared_error(predict_y,test_y))#,squared = False))
    errors_baseline = (mean_squared_error(predict_y_base,test_y))#,squared = False))

    results = [errors_Grid_CV,errors_Random_CV,errors_baseline]
    print('ridge results:',results)

    if True:
        fig=plt.figure(figsize=(15,8))
        x_axis = range(3)

        plt.bar(x_axis, results)
        plt.xticks(x_axis, ('GridSearchCV','RandomizedSearchCV', 'Baseline'))
        #plt.show()
        plt.savefig('ridge_error_compare.png')

        print('min index:',results.index(min(results)))

        if results.index(min(results)) is 0:
            model = ridge_grid
            pred_y = predict_y_grid
        else:
            if results.index(min(results)) is 1:
                model = ridge_random
                pred_y = predict_y_random
            else:
                mode = ridge
                pred_y = predict_y_base

        #feature importance
        #predictors = x_train.columns
        #coef = Series(lreg.coef_,predictors).sort_values()
        #coef.plot(kind='bar', title='Model Coefficients, kflod:'+str(kfold))

        #plt.show()
        #plt.savefig('ridge_feature_importance.png')

        fig=plt.figure(figsize=(20,8))
        ax = fig.gca()
        x_label = range(0,len(pred_y))
        plt.title("kfold="+str(kfold))
        ax.plot(x_label, pred_y, 'r--', label = "predict")
        ax.plot(x_label, test_y, label = "ground_truth")
        ax.set_ylim(0, 200)
        ax.legend()
        #plt.show()
        plt.savefig('ridge_prediction.png')

    return ridge_grid.predict,ridge_grid.best_estimator_
