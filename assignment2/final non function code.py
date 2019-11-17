import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime
from sklearn.svm import SVR

#start runtime clock
start=datetime.now()
#set reproducible seed



raw = pd.read_csv('assignment2/forestfires.csv')

# subset X columns and Y column
X = raw.iloc[:, 0:12]
y = np.array(raw.iloc[:, 12])


# cos sin transform categorical sequential features
# Encode Data
monthDict = dict(zip(['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'],[1,2,3,4,5,6,7,8,9,10,11,12]))
dayDict = dict(zip(['mon','tue','wed','thu','fri','sat','sun'],[1,2,3,4,5,6,7]))

X['month'] = X['month'].map(monthDict).astype(float)
X['day'] = X['day'].map(dayDict).astype(float)


# map each cyclical variable onto a circle so lowest value for that variable appears right next to the largest value.
X['day_sin'] = np.sin(X.day*(2.*np.pi/7))
X['day_cos'] = np.cos(X.day*(2.*np.pi/7))
X['mnth_sin'] = np.sin(X.month*(2.*np.pi/12))
X['mnth_cos'] = np.cos(X.month*(2.*np.pi/12))



# drop original cat ONLY USE FOR cos sin transform
X = X.drop(['month', 'day'], 1)

# fix int warning
X['X'] = X['X'].astype(float)
X['Y'] = X['Y'].astype(float)
X['RH'] = X['RH'].astype(float)


# define negative log likelihood of sample
def negative_log_likelihood(y, p):
    result = 0.5 * np.log(2 * np.pi * np.var(p)) + (((y - np.mean(p)) ** 2) / (2 * np.var(p)))
    return result


#feature selection list
STFWIM = ['X', 'Y', 'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain',
       'day_sin', 'day_cos', 'mnth_sin', 'mnth_cos']

STFWI = ['X', 'Y', 'FFMC', 'DMC', 'DC', 'ISI', 'day_sin', 'day_cos', 'mnth_sin', 'mnth_cos']

STM = ['X', 'Y', 'temp', 'RH', 'wind', 'rain', 'day_sin', 'day_cos', 'mnth_sin', 'mnth_cos']

FWI = ['FFMC', 'DMC', 'DC', 'ISI']

M = ['temp', 'RH', 'wind', 'rain']

#set hyper paramaters
parameters =[{"estimator__regressor__penalty": [None]},
             {"estimator__regressor__penalty": ['l2', 'l1'],"estimator__regressor__alpha": [1e-5,1e-4,1e-3,1e-2,1e-1]},
             {"estimator__regressor__penalty": ['elasticnet'],"estimator__regressor__alpha": [1e-5,1e-4,1e-3,1e-2,1e-1],
              "estimator__regressor__l1_ratio": [0.001, 0.25, 0.5, 0.75, 0.999]}]

# build pipeline
pipeline = Pipeline([ ('scaler', MinMaxScaler()),
                     ('estimator', TransformedTargetRegressor(regressor=SGDRegressor(max_iter=5, tol=-np.infty, random_state=42)
                                                              , func=np.log1p, inverse_func=np.expm1))])

# define outer and inner folds
outer_kv = KFold(n_splits=10, shuffle=True, random_state=42)
inner_kv = KFold(n_splits=10, shuffle=True, random_state=42)

cv = GridSearchCV(estimator=pipeline, param_grid=parameters, cv=inner_kv, iid=True,
                  scoring= "neg_mean_squared_error", n_jobs=-1, verbose=True)

feature_space = [STFWIM, STFWI, STM, FWI, M]
inner_result = []
outer_result = []
saving_inner_results = []
i = 0
for train, test in outer_kv.split(X):
    print("run", i)
    #loop over feature space using only training data and cross validator hyper param tuner
    for f in feature_space:
        cv.fit(X.loc[X.index[train],f], y[train] )
        #persist models to fit best on training set
        inner_result.append([cv,cv.best_params_,f,cv.best_score_])
        print(f)
        print(cv.best_score_)
    #persits and reset inner result for next fold
    inner_df = pd.DataFrame(inner_result)
    #reset inner result
    inner_result = []
    # receive best model of run to fit on test set
    best_params_arg = inner_df.loc[:, 3].argmax()
    best_params = inner_df.iloc[best_params_arg,:]
    # fit best cv model hyper parameters on best feature set for that fold
    bcv = best_params[0]
    bfs = best_params[2]
    bcv.fit(X.loc[X.index[train],bfs], y[train])

    # get training/val and test scores
    train_score = best_params[3]
    test_score = bcv.score(X.loc[X.index[test],bfs], y[test])

    # get predictions and retrieve nll of test folds
    y_preds = cv.predict(X.loc[X.index[test],bfs])
    mae = mean_absolute_error(y[test], y_preds)
    nllval = negative_log_likelihood(y[test], y_preds)

    outer_result.append([i,train_score,test_score, mae, bfs,  best_params[1], y[test], nllval])
    i += 1

testing = pd.DataFrame(outer_result)
testing.columns = ['fold_number','train_nmse','test_nmse','test_mae','best_feature_set','best_hyperparams','test_set','nll']

SGDRegressor(max_iter=5, tol=-np.infty, random_state=42)

datetime.now()-start



def nested_crossval(model , parameters):
    # build pipeline
    pipeline = Pipeline([('scaler', MinMaxScaler()),
                         ('estimator',
                          TransformedTargetRegressor(regressor=model
                                                     , func=np.log1p, inverse_func=np.expm1))])

    # define outer and inner folds
    outer_kv = KFold(n_splits=10, shuffle=True, random_state=42)
    inner_kv = KFold(n_splits=10, shuffle=True, random_state=42)

    cv = GridSearchCV(estimator=pipeline, param_grid=parameters, cv=inner_kv, iid=True,
                      scoring="neg_mean_squared_error", n_jobs=-1, verbose=True)

    feature_space = [STFWIM, STFWI, STM, FWI, M]
    inner_result = []
    outer_result = []
    saving_inner_results = []
    i = 0
    for train, test in outer_kv.split(X):
        print("run", i)
        # loop over feature space using only training data and cross validator hyper param tuner
        for f in feature_space:
            cv.fit(X.loc[X.index[train], f], y[train])
            # persist models to fit best on training set
            inner_result.append([cv, cv.best_params_, f, cv.best_score_])
            print(f)
            print(cv.best_score_)
        # persits and reset inner result for next fold
        inner_df = pd.DataFrame(inner_result)
        # reset inner result
        inner_result = []
        # receive best model of run to fit on test set
        best_params_arg = inner_df.loc[:, 3].argmax()
        best_params = inner_df.iloc[best_params_arg, :]
        # fit best cv model hyper parameters on best feature set for that fold
        bcv = best_params[0]
        bfs = best_params[2]
        bcv.fit(X.loc[X.index[train], bfs], y[train])

        # get training/val and test scores
        train_score = best_params[3]
        test_score = bcv.score(X.loc[X.index[test], bfs], y[test])

        # get predictions and retrieve nll of test folds
        y_preds = cv.predict(X.loc[X.index[test], bfs])
        mae = mean_absolute_error(y[test], y_preds)
        nllval = negative_log_likelihood(y[test], y_preds)
        mean_nll = np.mean(nllval)

        outer_result.append([i, train_score, test_score, mae, bfs, best_params[1], y[test], nllval, mean_nll])
        i += 1

    testing = pd.DataFrame(outer_result)
    testing.columns = ['fold_number', 'train_nmse', 'test_nmse', 'test_mae', 'best_feature_set', 'best_hyperparams',
                       'test_set', 'nll', 'mean_nll']

    return testing


nested_crossval(SGDRegressor(max_iter=5, tol=-np.infty, random_state=42),
                              [{"estimator__regressor__penalty": [None]},
                               {"estimator__regressor__penalty": ['l2', 'l1'],
                                "estimator__regressor__alpha": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]},
                               {"estimator__regressor__penalty": ['elasticnet'],
                                "estimator__regressor__alpha": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
                                "estimator__regressor__l1_ratio": [0.001, 0.25, 0.5, 0.75, 0.999]}])

print("average test fold RMSE:", np.sqrt(np.mean(np.negative(svr_results['test_nmse']))))
print("average test fold MSE:",np.mean(np.negative(svr_results['test_nmse'])))
print("average test fold MAE (MAD):", np.mean(svr_results['test_mae']))
print("average test fold MAE (MAD):", np.mean(svr_results['mean_nll']))


def nested_crossval(model, parameters):
    # build pipeline
    pipeline = Pipeline([('scaler', MinMaxScaler()),
                         ('estimator',
                          TransformedTargetRegressor(regressor=model
                                                     , func=np.log1p, inverse_func=np.expm1))])

    # define outer and inner folds
    outer_kv = KFold(n_splits=10, shuffle=True, random_state=42)
    inner_kv = KFold(n_splits=10, shuffle=True, random_state=42)

    # instantiate inner CV grid search
    cv = GridSearchCV(estimator=pipeline, param_grid=parameters, cv=inner_kv,
                      scoring="neg_mean_squared_error", n_jobs=-1, verbose=True)

    # create list of feature spaces to search and list of result to persist
    feature_space = [STFWIM, STFWI, STM, FWI, M]
    inner_result = []
    outer_result = []
    saving_inner_results = []
    i = 0
    for train, test in outer_kv.split(X):
        print("run", i)
        # loop over feature space using only training data and cross validator hyper param tuner
        for f in feature_space:
            cv.fit(X.loc[X.index[train], f], y[train])
            # persist models to fit best on training set
            inner_result.append([cv, cv.best_params_, f, cv.best_score_])
            print(f)
            print(cv.best_score_)
        # persits and reset inner result for next fold
        inner_df = pd.DataFrame(inner_result)
        # reset inner result
        inner_result = []
        # receive best model of run to fit on test set
        best_params_arg = inner_df.loc[:, 3].argmax()
        best_params = inner_df.iloc[best_params_arg, :]
        # fit best cv model hyper parameters on best feature set for that fold
        bcv = best_params[0]
        bfs = best_params[2]
        bcv.fit(X.loc[X.index[train], bfs], y[train])

        # get training/val and test scores
        train_score = best_params[3]
        test_score = bcv.score(X.loc[X.index[test], bfs], y[test])

        # get predictions and retrieve nll of test folds
        y_preds = cv.predict(X.loc[X.index[test], bfs])
        mae = mean_absolute_error(y[test], y_preds)
        nllval = negative_log_likelihood(y[test], y_preds)
        mean_nll = np.mean(nllval)

        outer_result.append([i, train_score, test_score, mae, bfs, best_params[1], y[test], nllval, mean_nll])
        i += 1

    testing = pd.DataFrame(outer_result)
    testing.columns = ['fold_number', 'train_nmse', 'test_nmse', 'test_mae', 'best_feature_set', 'best_hyperparams',
                       'test_set', 'nll', 'mean_nll']

    return testing





for train, test in outer_kv.split(X):
    print("run", i)
    #loop over feature space using only training data and cross validator hyper param tuner
    for f in feature_space:
        cv.fit(X.loc[X.index[train],f], y[train] )
        #persist models to fit best on training set
        inner_result.append([cv,cv.best_params_,f,cv.best_score_])
        print(f)
        print(cv.best_score_)
    #persits and reset inner result for next fold
    inner_df = pd.DataFrame(inner_result)
    #reset inner result
    inner_result = []
    # receive best model of run to fit on test set
    best_params_arg = inner_df.loc[:, 3].argmax()
    best_params = inner_df.iloc[best_params_arg,:]
    # fit best cv model hyper parameters on best feature set for that fold
    bcv = best_params[0]
    bfs = best_params[2]
    bcv.fit(X.loc[X.index[train],bfs], y[train])

    # get training/val and test scores
    train_score = best_params[3]
    test_score = bcv.score(X.loc[X.index[test],bfs], y[test])

    # get predictions and retrieve nll of test folds
    y_preds = cv.predict(X.loc[X.index[test],bfs])
    mae = mean_absolute_error(y[test], y_preds)
    nllval = negative_log_likelihood(y[test], y_preds)

    outer_result.append([i,train_score,test_score, mae, bfs,  best_params[1], y[test], nllval])
    i += 1

testing = pd.DataFrame(outer_result)










