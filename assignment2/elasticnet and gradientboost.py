import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_validate
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pd.options.mode.chained_assignment = None

raw = pd.read_csv('assignment2/forestfires.csv')

# subset X columns and Y column
X = raw.iloc[:, 0:12]
# log transform
y = np.log(np.array(raw.iloc[:, 12]) + 1)
# y = np.array(raw.iloc[:, 12]) NO NEED TO USE

# create random train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.001, random_state=42)

# cos sin transform categorical sequential features
# Encode Data
monthDict = dict(zip(['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'],[0,1,2,3,4,5,6,7,8,9,10,11]))
dayDict = dict(zip(['mon','tue','wed','thu','fri','sat','sun'],[0,1,2,3,4,5,6]))

X_train['month'] = X_train['month'].map(monthDict).astype(float)
X_train['day'] = X_train['day'].map(dayDict).astype(float)

X_test['month'] = X_test['month'].map(monthDict).astype(float)
X_test['day'] = X_test['day'].map(dayDict).astype(float)


# map each cyclical variable onto a circle so lowest value for that variable appears right next to the largest value.
X_train['day_sin'] = np.sin(X_train.day*(2.*np.pi/7))
X_train['day_cos'] = np.cos(X_train.day*(2.*np.pi/7))
X_train['mnth_sin'] = np.sin(X_train.month*(2.*np.pi/12))
X_train['mnth_cos'] = np.cos(X_train.month*(2.*np.pi/12))

X_test['day_sin'] = np.sin(X_test.day*(2.*np.pi/7))
X_test['day_cos'] = np.cos(X_test.day*(2.*np.pi/7))
X_test['mnth_sin'] = np.sin(X_test.month*(2.*np.pi/12))
X_test['mnth_cos'] = np.cos(X_test.month*(2.*np.pi/12))


# drop original cat ONLY USE FOR cos sin transform
X_train = X_train.drop(['month', 'day'], 1)
X_test = X_test.drop(['month', 'day'], 1)

# attempt using only sensor data
X_train = X_train[['temp', 'RH', 'wind', 'rain']]
X_test = X_test[['temp', 'RH', 'wind', 'rain']]

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

# scale features
scaler = MinMaxScaler()

scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# standardise features
#scaler = StandardScaler()

#scaler.fit(X_train)
#X_train = scaler.transform(X_train)
#X_test = scaler.transform(X_test)

# perform dimension reduction through PCA
#X_train.shape

#pca = PCA(n_components=1, svd_solver='full')
#pca.fit(X_train)

#X_train_pca = pca.transform(X_train)
#X_test_pca = pca.transform(X_test)



# modelling and cross validation due to small samples
lnr = SGDRegressor(max_iter=100, tol=1e-3)
lr = LinearRegression()
bst = XGBRegressor(objective='reg:squarederror', silent=0, min_child_weight= 10, eta= 0.01, subsample=0.5, max_depth=3)


# turn mse from metric into scorer
scores = cross_validate(lnr,
                        X_train,
                        y_train,
                        cv=10,
                        return_train_score=True,
                        scoring='neg_mean_squared_error',
                        n_jobs=16)  # you wil want to change this to your pc number of threads
print("elastic test", np.mean(scores["test_score"]), "\n elastic train", np.mean(scores["train_score"]))



scores = cross_validate(lr,
                        X_train,
                        y_train,
                        cv=10,
                        return_train_score=True,
                        scoring='neg_mean_squared_error',
                        n_jobs=16)  # you wil want to change this to your pc number of threads
print("lr test", np.mean(scores["test_score"]), "\n lr train", np.mean(scores["train_score"]))





scores = cross_validate(bst,
                        X_train,
                        y_train,
                        cv=10,
                        return_train_score=True,
                        scoring='neg_mean_squared_error',
                        n_jobs=16)  # you wil want to change this to your pc number of threads
print("boosted test", np.mean(scores["test_score"]), "\n boosted train", np.mean(scores["train_score"]))
