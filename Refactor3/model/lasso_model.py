from sklearn.model_selection import PredefinedSplit
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
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


def lasso(X, Y, kfold=3, feature_set=None):

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
    lasso = Lasso(random_state = 42)
    lasso.fit(train_X, train_y)
    print('Base Parameters in use:')
    print(lasso.get_params())


    # grid search

    alpha_log = np.linspace(-8,5,14)

    alpha = []
    for i in alpha_log:
        a = math.pow(10,i)
        alpha = alpha + [a]

    grid_grid = {'alpha': alpha}
    
    print(grid_grid)
    lasso_grid = GridSearchCV(estimator=lasso, param_grid=grid_grid, scoring='neg_mean_squared_error', cv = ps2.split(), verbose=2, n_jobs=-1)

    # Fit the grid search model
    lasso_grid.fit(train_X, train_y)
    BestPara_grid = lasso_grid.best_params_
    print("grid search, best parameter:", lasso_grid.best_params_)
    #cv_results_grid = lasso_grid.cv_results_


    lr_unit =  BestPara_grid['alpha']/10
    alpha = [x for x in np.linspace(start = lr_unit, stop = lr_unit*99, num = 99)]

    grid_grid = {'alpha': alpha}
    lasso_grid2 = GridSearchCV(estimator=lasso, param_grid=grid_grid, scoring='neg_mean_squared_error', cv = ps2.split(), verbose=2, n_jobs=-1)

    # Fit the grid search model
    lasso_grid2.fit(train_X, train_y)
    print("grid search, best parameter:", lasso_grid2.best_params_)
    #cv_results_grid2 = lasso_grid2.cv_results_    


    #prediction
    predict_y_grid2 = lasso_grid2.predict(test_X)
    predict_y_grid = lasso_grid.predict(test_X)
    predict_y_base = lasso.predict(test_X)

    # Performance metrics
    errors_Grid2_CV = np.sqrt(mean_squared_error(predict_y_grid2,test_y))
    errors_Grid_CV = np.sqrt(mean_squared_error(predict_y_grid,test_y))
    errors_baseline = np.sqrt(mean_squared_error(predict_y_base,test_y))

    results = [errors_Grid2_CV, errors_Grid_CV,errors_baseline]
    print('lasso results:',results)

    if False:
        #feature importance
        num_feature = len(lasso_grid2.best_estimator_.coef_)
        plt.figure(figsize=(24,6))
        plt.bar(range(1,num_feature*4,4), lasso_grid2.best_estimator_.coef_)
        label_name = X.keys()
        plt.xticks(range(1,num_feature*4,4), label_name)
        plt.title("Lasso Feature Importances"+",kfold="+str(kfold))
        plt.show()
        #(lasso_grid2.best_estimator_.coef_)
        #print(label_name)        
        
    return lasso_grid2, results, lasso_grid2.best_params_
