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
from xgboost import XGBRegressor
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
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

#passing coef amd importance through pipe and TransformedTargetRegressor
class MyPipeline(Pipeline):
    @property
    def coef_(self):
        return self._final_estimator.coef_
    @property
    def feature_importances_(self):
        return self._final_estimator.feature_importances_

class MyTransformedTargetRegressor(TransformedTargetRegressor):
    @property
    def feature_importances_(self):
        return self.regressor_.feature_importances_

    @property
    def coef_(self):
        return self.regressor_.coef_


# build pipeline
pipeline = MyPipeline([ ('scaler', MinMaxScaler()),
                     ('estimator', MyTransformedTargetRegressor(regressor=SGDRegressor(), func=np.log1p, inverse_func=np.expm1))])

# define tuning grid
parameters = {"estimator__estimator__regressor__alpha": [1e-5,1e-4,1e-3,1e-2,1e-1],
              "estimator__estimator__regressor__l1_ratio": [0.001,0.25,0.5,0.75,0.999]}

# define outer and inner folds
outer_kv = KFold(n_splits=10, shuffle=True, random_state=None)
inner_kv = KFold(n_splits=10, shuffle=True, random_state=42)
rfcv = RFECVCoef(estimator=pipeline, step=1, cv=inner_kv, scoring="neg_mean_squared_error")

cv = GridSearchCV(estimator=rfcv, param_grid=parameters, cv=inner_kv, iid=True,
                  scoring= "neg_mean_squared_error", n_jobs=-1, verbose=True)

mse_train = []
mse_test = []
nll = []
for train, test in outer_kv.split(X):
    # fit inner k fold loop
    cv.fit(X.iloc[train,:], y[train] )

    # get training/val and test scores
    train_score = cv.best_score_
    test_score = cv.score(X.iloc[test], y[test])

    # get predictions and retrieve nll of test folds
    y_preds = cv.predict(X.iloc[test])
    nll.append(negative_log_likelihood(y[test], y_preds))

    mse_train.append(train_score)
    mse_test.append(test_score)

np.mean(np.negative(np.array(mse_test)))
np.sqrt(np.mean(np.negative(np.array(mse_test))))

print("best hyperparameters for linear regression:",cv.best_params_)
cv.best_estimator_.coef_
cv.best_estimator_.steps[1][1].

dir(cv)
cv.cv_results_



# build regression pipeline
pipeline = Pipeline([ ('scaler', MinMaxScaler()),
                     ('estimator', TransformedTargetRegressor(regressor=XGBRegressor(), func=np.log1p, inverse_func=np.expm1))]) #
# define tuning grid
parameters = {"estimator__regressor__learning_rate": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ],
              "estimator__regressor__max_depth": [3, 4, 5, 6, 8, 10, 12, 15],
              "estimator__regressor__min_child_weight": [1, 3, 5, 7 ],
              "estimator__regressor__gamma": [0.0, 0.1, 0.2 , 0.3, 0.4 ],
              "estimator__regressor__colsample_bytree": [0.3, 0.4, 0.5 , 0.7 ]}



# define outer and inner folds
outer_kv = KFold(n_splits=5, shuffle=True, random_state=None)
inner_kv = KFold(n_splits=5, shuffle=True, random_state=None)

# instantiate inner cv
#rfcv = RFECV(estimator=pipeline, step=1, cv=inner_kv, scoring="neg_mean_squared_error")

cv = GridSearchCV(estimator=pipeline, param_grid=parameters, cv=inner_kv, iid=True,
                  scoring= "neg_mean_squared_error", n_jobs=-1, verbose=True)

mse_train = []
mse_test = []
nll = []
for train, test in outer_kv.split(X):
    # fit inner k fold loop
    cv.fit(X.iloc[train,:], y[train] )

    # get training/val and test scores
    train_score = cv.best_score_
    test_score = cv.score(X.iloc[test], y[test])

    # get predictions and retrieve nll of test folds
    y_preds = cv.predict(X.iloc[test])
    nll.append(negative_log_likelihood(y[test], y_preds))

    mse_train.append(train_score)
    mse_test.append(test_score)

np.mean(np.negative(np.array(mse_test)))
np.sqrt(np.mean(np.negative(np.array(mse_test))))

print("best hyperparameters for linear regression:",cv.best_params_)















