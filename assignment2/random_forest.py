import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

raw = pd.read_csv('assignment2/forestfires.csv')

# subset X columns and Y column
X = raw.iloc[:, 0:12]
# log transform
y = np.log(np.array(raw.iloc[:, 12]) + 1)

X.shape
y.shape

raw['area']

# create random train test split 80:20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.001, random_state=42)

# one hot encode categorical variables
# consider using seasons instead of months to reduce cardinality
categorical_mask = X_train.dtypes == object
categorical_columns = X_train.columns[categorical_mask]

# fit encoder
enc = OneHotEncoder(sparse=False)
enc.fit(X_train[categorical_columns])

# transform cat vars
X_train_cat = enc.transform(X_train[categorical_columns])
X_test_cat = enc.transform(X_test[categorical_columns])

# merge one hot with X as array
X_train = X_train.drop(X_train.columns[[0, 1, 2, 3]], axis=1)
X_test = X_test.drop(X_test.columns[[0, 1, 2, 3]], axis=1)

X_train = np.concatenate((np.array(X_train), X_train_cat), axis=1)
X_test = np.concatenate((np.array(X_test), X_test_cat), axis=1)

# scale features
scaler = MinMaxScaler()

scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# modelling and cross validation due to small samples
forest = RandomForestRegressor()

# turn mse from metric into scorer
scores = cross_validate(forest,
                        X_train,
                        y_train,
                        cv=5,
                        return_train_score=True,
                        scoring='neg_mean_squared_error',
                        n_jobs=16)  # you wil want to change this to your pc number of threads
print(scores)


