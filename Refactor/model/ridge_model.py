from sklearn.model_selection import PredefinedSplit
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import math
import numpy as np
import matplotlib.pyplot as plt



def index_splitter(N, fold):
    index_split = []
    test_num = int(N/fold)
    train_num = N-test_num

    for i in range(0,train_num):
        index_split.append(-1)

    for i in range(train_num,N):
        index_split.append(0)

    return index_split


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

    # base
    ridge = Ridge(random_state = 42)
    ridge.fit(train_X, train_y)
    print('Base Parameters in use:')
    print(ridge.get_params())

    # grid search
    alpha_log = np.linspace(-8,5,14)

    alpha = []
    for i in alpha_log:
        a = math.pow(10,i)
        alpha = alpha + [a]

    grid_grid = {'alpha': alpha}
    ridge_grid = GridSearchCV(estimator=ridge, param_grid=grid_grid, scoring='neg_mean_squared_error', cv = ps2.split(), verbose=2, n_jobs=-1)

    # Fit the grid search model
    ridge_grid.fit(train_X, train_y)
    BestPara_grid = ridge_grid.best_params_
    print("grid search, best parameter:", ridge_grid.best_params_)
    cv_results_grid = ridge_grid.cv_results_
    
    new_measure =  int(ridge_grid.best_params_['alpha'])
    new_alpha = [x for x in range(100,new_measure*10,10)]
    print(new_alpha)
    grid_grid = {'alpha': new_alpha}
    ridge_grid2 = GridSearchCV(estimator=ridge, param_grid=grid_grid, scoring='neg_mean_squared_error', cv = ps2.split(), verbose=2, n_jobs=-1)

    # Fit the grid search model
    ridge_grid2.fit(train_X, train_y)
    BestPara_grid = ridge_grid2.best_params_
    print("grid search, best parameter:", ridge_grid2.best_params_)
    cv_results_grid2 = ridge_grid2.cv_results_    
    

    #prediction
    predict_y_grid = ridge_grid.predict(test_X)
    predict_y_base = ridge.predict(test_X)
    predict_y_grid2 = ridge_grid2.predict(test_X)

    # Performance metrics
    errors_Grid_CV = np.sqrt(mean_squared_error(predict_y_grid,test_y))
    errors_Grid2_CV = np.sqrt(mean_squared_error(predict_y_grid2,test_y))
    errors_baseline = np.sqrt(mean_squared_error(predict_y_base,test_y))

    results = [errors_Grid2_CV,errors_Grid_CV,errors_baseline]
    print('ridge results:',results)


    return ridge_grid2
