import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import KFold
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from scipy import special
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.base import BaseEstimator
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime
#start runtime clock
start=datetime.now()


pd.options.mode.chained_assignment = None

raw = pd.read_csv('assignment2/forestfires.csv')

# subset X columns and Y column
X = raw.iloc[:, 0:12]
y = np.array(raw.iloc[:, 12])


# cos sin transform categorical sequential features
# Encode Data
monthDict = dict(zip(['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'],[0,1,2,3,4,5,6,7,8,9,10,11]))
dayDict = dict(zip(['mon','tue','wed','thu','fri','sat','sun'],[0,1,2,3,4,5,6]))

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

# attempt using only sensor data
#X = X[['temp', 'RH', 'wind', 'rain']]

# create random train test split
#X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=.15)


# one hot encode categorical variables
# categorical_mask = X_train.dtypes == object
# categorical_columns = X_train.columns[categorical_mask]

# fit encoder
# enc = OneHotEncoder(sparse=False)
# enc.fit(X_train[categorical_columns])

# transform cat vars
# X_train_cat = enc.transform(X_train[categorical_columns])
# X_test_cat = enc.transform(X_test[categorical_columns])

# merge one hot with X as array
# X_train = X_train.drop(X_train.columns[[0, 1, 2, 3]], axis=1)
# X_test = X_test.drop(X_test.columns[[0, 1, 2, 3]], axis=1)

# X_train = np.concatenate((np.array(X_train), X_train_cat), axis=1)
# X_test = np.concatenate((np.array(X_test), X_test_cat), axis=1)

# define negative log likelihood of sample
def negative_log_likelihood(y, p):
    result = -0.5 * np.log(2 * np.pi * np.var(p)) + (((y - np.mean(p)) ** 2) / 2 * np.var(p))
    return result


#feature selection list
STFWIM = ['X', 'Y', 'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain',
       'day_sin', 'day_cos', 'mnth_sin', 'mnth_cos']

STFWI = ['X', 'Y', 'FFMC', 'DMC', 'DC', 'ISI', 'day_sin', 'day_cos', 'mnth_sin', 'mnth_cos']

STM = ['X', 'Y', 'temp', 'RH', 'wind', 'rain', 'day_sin', 'day_cos', 'mnth_sin', 'mnth_cos']

FWI = ['FFMC', 'DMC', 'DC', 'ISI']

M = ['temp', 'RH', 'wind', 'rain']

# build pipeline
pipeline = Pipeline([ ('scaler', MinMaxScaler()),
                     ('estimator', TransformedTargetRegressor(regressor=SGDRegressor(max_iter=5, tol=-np.infty), func=np.log1p, inverse_func=np.expm1))])

# define tuning grid
parameters = {"estimator__regressor__alpha": [1e-5,1e-4,1e-3,1e-2,1e-1],
              "estimator__regressor__l1_ratio": [0.001,0.25,0.5,0.75,0.999]}

# define outer and inner folds
outer_kv = KFold(n_splits=10, shuffle=True, random_state=42)
inner_kv = KFold(n_splits=10, shuffle=True, random_state=42)

cv = GridSearchCV(estimator=pipeline, param_grid=parameters, cv=inner_kv, iid=True,
                  scoring= "neg_mean_squared_error", n_jobs=-1, verbose=True)

feature_space = [STFWI, STM, FWI, M]
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
    #saving_inner_result.append
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

print("RMSE:", np.sqrt(np.mean(np.negative(testing['test_nmse']))))
print("MSE:",np.mean(np.negative(testing['test_nmse'])))
print("MAE:", np.mean(testing['test_mae']))


print(datetime.now()-start)

testing.to_csv("assignment2/lr_results_set.csv")