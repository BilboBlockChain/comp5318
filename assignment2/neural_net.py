from tensorflow.keras import Input
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate

raw = pd.read_csv('assignment2/forestfires.csv')

# subset X columns and Y column
X = raw.iloc[:, 0:12]
# log transform
y = np.log(np.array(raw.iloc[:, 12]) + 1).reshape(y.shape[0], 1)

# create random train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.001, random_state=42)

# Encode Data
X_train.month.replace(('jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'),(1,2,3,4,5,6,7,8,9,10,11,12), inplace=True)
X_train.day.replace(('mon','tue','wed','thu','fri','sat','sun'),(1,2,3,4,5,6,7), inplace=True)

# scale features
scaler = MinMaxScaler()

scaler.fit(X_train)
X_train = scaler.transform(X_train)
#X_test = scaler.transform(X_test)

#model building 

input_layer = layers.Input(shape = (X_train.shape[1],))
hidden1 = layers.Dense(10, activation = 'relu')(input_layer)
hidden2 = layers.Dense(3, activation = 'relu')(hidden1)
output_layer = layers.Dense(1, activation = 'relu')(hidden2)

model = Model(inputs = input_layer, outputs = output_layer)
model.summary()

model.compile(optimizer = 'adam',
              loss = 'mean_squared_error', 
              metrics = ['mean_squared_error'])

dl = KerasRegressor(build_fn=model, nb_epoch=600, batch_size=5, verbose=0)
estimator = KerasRegressor(build_fn=model, nb_epoch=600, batch_size=5, verbose=0)

kfold = KFold(n_splits=30, random_state=42)
results = cross_val_score(estimator, X_train, y_train, cv=kfold)



#
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout

def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(10, input_dim=X_train.shape[1], kernel_initializer='uniform', activation='relu'))
    #model.add(Dropout(0.2))
    #model.add(Dense(5, kernel_initializer='uniform', activation='relu', kernel_constraint=maxnorm(3)))
    #model.add(Dropout(0.2))
    model.add(Dense(3, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(1, kernel_initializer='uniform', activation='relu'))
    
    # compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# evaluate model with standardized dataset
estimator = KerasRegressor(build_fn=baseline_model, nb_epoch=1600, batch_size=1, verbose=0)

kfold = KFold(n_splits=30, random_state=seed)
results = cross_val_score(estimator, X_train, y_train, cv=kfold)
print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))

