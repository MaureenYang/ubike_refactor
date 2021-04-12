# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 20:56:50 2019

@author: Mandy

"""

from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import PredefinedSplit
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


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
    alpha = np.linspace(0,1,10)
    random_grid = {'alpha': alpha}

    ridge = Ridge(random_state = 42)

    from pprint import pprint

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

    ## Grid search of parameters, using 3 fold cross validation based on Random search
    from sklearn.model_selection import GridSearchCV

    # Number of trees in random forest
    alpha = [int(x) for x in range(BestPara_random["alpha"]-2,BestPara_random["alpha"]+2,100)]

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
    from sklearn.metrics import mean_squared_log_error
    from sklearn.metrics import mean_squared_error

    def RMLSE(predict_y_grid, predict_y, predict_y_base, test_y):
        errors_Grid_CV = np.sqrt(mean_squared_log_error(predict_y_grid,test_y))
        errors_Random_CV = np.sqrt(mean_squared_log_error(predict_y,test_y))
        errors_baseline = np.sqrt(mean_squared_log_error(predict_y_base,test_y))
        return errors_Grid_CV, errors_Random_CV, errors_baseline

    errors_Grid_CV = (mean_squared_error(predict_y_grid,test_y,squared = False))
    errors_Random_CV = (mean_squared_error(predict_y,test_y,squared = False))
    errors_baseline = (mean_squared_error(predict_y_base,test_y,squared = False))

    x_axis = range(3)
    results = [errors_Grid_CV,errors_Random_CV,errors_baseline]
    plt.bar(x_axis, results)
    plt.xticks(x_axis, ('GridSearchCV','RandomizedSearchCV', 'Baseline'))
    plt.show()

# =============================================================================
#     #feature importance
#     num_feature = len(rf_random.best_estimator_.feature_importances_)
#     plt.figure(figsize=(12,6))
#     plt.bar(range(0,num_feature*4,4),rf_random.best_estimator_.feature_importances_)
#     label_name = X.keys()
#     plt.xticks(range(0,num_feature*4,4), label_name)
#     plt.title("Feature Importances")
#     plt.show()
# =============================================================================

    #feature importance
    num_feature = len(ridge_grid.best_estimator_.feature_importances_)
    plt.figure(figsize=(24,6))
    plt.bar(range(0,num_feature*4,4),ridge_grid.best_estimator_.feature_importances_)

    label_name = X.keys()

    plt.xticks(range(0,num_feature*4,4), label_name)
    plt.title("Feature Importances"+",kfold="+str(kfold))
    plt.show()
    fig=plt.figure(figsize=(20,8))
    ax = fig.gca()
    x_label = range(0,len(predict_y_grid))
    plt.title("kfold="+str(kfold))
    ax.plot(x_label, predict_y_grid, 'r--', label = "predict")
    ax.plot(x_label, test_y, label = "ground_truth")
    ax.set_ylim(0, 200)
    ax.legend()
    plt.show()

    return ridge_grid.predict,ridge_grid.best_estimator_
