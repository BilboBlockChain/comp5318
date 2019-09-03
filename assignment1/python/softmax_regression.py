import h5py
import numpy as np
import os

print(os.listdir("./data/train"))

with h5py.File('./data/train/images_training.h5','r') as H:
    data_train = np.copy(H['datatrain'])
with h5py.File('./data/train/labels_training.h5','r') as H:
    label_train = np.copy(H['labeltrain'])

# using H['datatest'], H['labeltest'] for test dataset.

print(data_train.shape,label_train.shape)

import matplotlib.pyplot as plt
data_train = data_train.reshape((data_train.shape[0], 28, 28))
plt.imshow(data_train[0], cmap=plt.get_cmap('gray'))
plt.title("class " + str(label_train[0]) + ": T-shirt/Top" )
plt.show()

#flatten tensor and create val set
flat_train = data_train.reshape(30000, 28 * 28)
flat_val = flat_train[20000:,]
flat_partial_train = flat_train[:20000,]

#data matrix already contained between 0 and 1 no transformation needed
np.min(flat_train)
np.max(flat_train)

#one hot encode y for softmax output 
hot_train = np.zeros((label_train.size, label_train.max()+1))
hot_train[np.arange(label_train.size), label_train] = 1

#create val labels
hot_val = hot_train[20000:,]
hot_partial_train = hot_train[:20000,]

#create softmax function 
def softmax(x):
    return np.exp(x)/ np.sum(np.exp(x), axis = 0)

#retrieve probabilites and arg max predictions
def outputs(X):
    probs = softmax(np.dot(X,w))
    preds = np.argmax(probs, axis = 1)
    return probs, preds


    



