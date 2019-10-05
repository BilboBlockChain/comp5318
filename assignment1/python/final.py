import h5py
import numpy as np
import os
import scipy as sp
import time
import pandas as pd

print(os.listdir("./assignment1/data/train"))

with h5py.File('./assignment1/data/train/images_training.h5','r') as H:
    data_train = np.copy(H['datatrain'])
with h5py.File('./assignment1/data/train/labels_training.h5','r') as H:
    label_train = np.copy(H['labeltrain'])

# using H['datatest'], H['labeltest'] for test dataset.

print(data_train.shape,label_train.shape)

import matplotlib.pyplot as plt
data_train = data_train.reshape((data_train.shape[0], 28, 28))
plt.imshow(data_train[0], cmap=plt.get_cmap('gray'))
plt.title("class " + str(label_train[0]) + ": T-shirt/Top" )
plt.show()


label_train[25000:,]


unique, counts = np.unique(label_train[25000:,], return_counts=True)
dict(zip(unique, counts))


t1 = time.time()
#create val set
data_val = data_train[25000:,]
partial_train = data_train[:25000,]




#one hot encode y for softmax output 
def oneHot(y):
    zeroesY = np.zeros((y.size, y.max() + 1))
    zeroesY[np.arange(y.size), y] = 1
    return zeroesY


#create val labels
label_val = oneHot(label_train[25000:,])
partial_label_train = oneHot(label_train[: 25000,])

#perform pca and reshape
#B is the training set so that all sets can use its mean and std for preproc
def pre_proc(A):
    flat_A = A.reshape(A.shape[0], A.shape[1] * A.shape[2])
#    flat_B = B.reshape(B.shape[0], B.shape[1] * B.shape[2])
#    Z = (flat_A - np.mean(flat_B, axis = 0))/np.std(flat_B, axis = 0)
    return flat_A

def svd_fit(A, comps):   
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    #the matrix XV = UD therefore only need right singular value
    V_tilde = Vt.T[:,0:comps]
    return V_tilde

###use if standardisation
#flat_partial_train = pre_proc(partial_train, partial_train)
#flat_data_val = pre_proc(data_val, partial_train)

flat_partial_train = pre_proc(partial_train)
flat_data_val = pre_proc(data_val)

#from sklearn import preprocessing
#scaler = preprocessing.StandardScaler().fit(flat_partial_train)

#flat_partial_train = scaler.transform(flat_partial_train)
#flat_data_val = scaler.transform(flat_data_val)

v = svd_fit(flat_partial_train, 28*28)


flat_partial_train = flat_partial_train.dot(v)
flat_data_val = flat_data_val.dot(v)

###rescale between 0 and 1
#flat_partial_train = (flat_partial_train/np.max(flat_partial_train, axis = 0))
#flat_data_val = (flat_data_val/np.max(flat_partial_train, axis = 0))


#create softmax function 
def softmax(Z): # Todo 
    e_Z = np.exp(Z)
    A = e_Z / e_Z.sum(axis = 1, keepdims = True)
    return A

#define softmax loss 
def softmax_loss(X, y, W, l):
    A = softmax(X.dot(W))
    n = len(X)
    return (-1/n) * np.sum(y * np.log(A)) + (l/2)*np.sum(W*W) 


#define softmax gradient 
def softmax_grad(X, y, W, l):
    A = softmax(X.dot(W))    # shape of (N, C)
    n = len(X) 
    return  ((np.dot(X.T,(A - y))/n)) + l*W



def softmax_fit(X, y, W, l, lr = 0.01, nepoches = 20,  batch_size = 1000, ):
    ep = 0 
    n = len(X)
    loss_hist = [softmax_loss(X, y, W, l)] # store history of loss 
    steps = int(np.ceil(n/batch_size))
    t0 = time.time()
    for ep in range(nepoches): 
        p_ids = np.random.permutation(n) # mix data 
        shuffle_x = X[p_ids]
        shuffle_y = y[p_ids]
        for i in range(steps):
            # get the i-th batch
            X_batch = shuffle_x[i:i + batch_size, :]
            y_batch = shuffle_y[i:i + batch_size]
            W -=  lr * softmax_grad(X_batch, y_batch, W, l)
        loss_hist.append(softmax_loss(X, y, W, l))
    t1 = time.time()
    runtime = t1 - t0
    print(f"internal loop speed for {nepoches} epochs and {batch_size} batch size is {runtime} seconds")
    return W, loss_hist

len(flat_partial_train)

W_init = np.random.randn(flat_partial_train.shape[1], len(partial_label_train[1]))
W, loss_hist = softmax_fit(flat_partial_train, 
                                      partial_label_train , 
                                      W_init, 
                                      nepoches = 750, 
                                      batch_size = 100,
                                      lr = 0.01, 
                                      l = .00)
    
t2 = time.time()
print(t2 - t1)


# predict fuction
def pred(W, X):
    A = softmax(X.dot(W))
    return np.argmax(A, axis = 1)

#accuracy
def accurancy(y_pre,y): # Todo 
    count = y_pre == y
    accurancy = count.sum()/len(count)
    return accurancy


y_pre = pred(W,flat_data_val)

y_pre.shape
label_train[: 25000,].shape
print("Accurancy of model on test set:",accurancy(y_pre,label_train[25000:,]))

results = pd.DataFrame({'label': label_train[25000:,], 'answer': label_train[25000:,] == y_pre})

props = results.groupby('label').answer.value_counts(normalize = True)
props
