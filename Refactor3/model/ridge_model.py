from sklearn.model_selection import PredefinedSplit
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

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

    # random search
    '''
    alpha = np.linspace(0.1,1,10)
    random_grid = {'alpha': alpha}
    ridge_random = RandomizedSearchCV(estimator=ridge, param_distributions=random_grid, scoring='neg_mean_squared_error',cv = ps2.split(), verbose=2, random_state=42, n_jobs=-1)

    # Fit the random search model
    ridge_random.fit(train_X, train_y)
    print("random search, best parameter:",ridge_random.best_params_)
    BestPara_random = ridge_random.best_params_
    cv_result_rd= ridge_random.cv_results_
    '''
    # grid search
    alpha = np.linspace(0.1,1,10)
    grid_grid = {'alpha': alpha}
    ridge_grid = GridSearchCV(estimator=ridge, param_grid=grid_grid, scoring='neg_mean_squared_error', cv = ps2.split(), verbose=2, n_jobs=-1)

    # Fit the grid search model
    ridge_grid.fit(train_X, train_y)
    BestPara_grid = ridge_grid.best_params_
    print("grid search, best parameter:", ridge_grid.best_params_)
    cv_results_grid = ridge_grid.cv_results_

    #prediction
    predict_y_grid = ridge_grid.predict(test_X)
    predict_y_base = ridge.predict(test_X)

    # Performance metrics
    errors_Grid_CV = (mean_squared_error(predict_y_grid,test_y))
    errors_baseline = (mean_squared_error(predict_y_base,test_y))

    results = [errors_Grid_CV,errors_baseline]
    print('ridge results:',results)

    if False:
        fig=plt.figure(figsize=(15,8))
        x_axis = range(3)

        plt.bar(x_axis, results)
        plt.xticks(x_axis, ('GridSearchCV','RandomizedSearchCV', 'Baseline'))
        #plt.show()
        plt.savefig('ridge_error_compare.png')

        print('min index:',results.index(min(results)))

        if results.index(min(results)) == 0:
            model = ridge_grid
            pred_y = predict_y_grid
        else:
            if results.index(min(results)) == 1:
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

    return ridge_grid.best_estimator_
