import h5py
import numpy as np
import os
import scipy as sp

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

v = svd_fit(flat_partial_train, 170)


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

#define softmax gradient 
#def softmax_grad(x, y, w):
#    A = softmax(x.dot(w))    # shape of (N, C)
#    id0 = range(x.shape[0])  # number of train data
#    A[id0, y] -= 1           # A - Y, shape of (N, C)
#    return x.T.dot(A)/x.shape[0]



#define softmax loss 
#def softmax_loss(x, y, w):
#    A = softmax(x.dot(w))
#    id0 = range(x.shape[0])
#    return -np.mean(np.log(A[id0, y]))

def softmax_loss(x, y, w, l):
    A = softmax(x.dot(w))
    n = x.shape[0] 
    return (-1/n) * np.sum(y * np.log(A)) + (l/2)*np.sum(w*w) 


#define softmax gradient 
def softmax_grad(x, y, w, l):
    A = softmax(x.dot(w))    # shape of (N, C)
    n = x.shape[0] 
    return  -((np.dot(x.T,(y - A))/n)) + l*w 


####other loop

#def softmax_fit(X, y, W, l, lr = 0.01, nepoches = 100, tol = 1e-5, batch_size = 1000):
#    W_old = W.copy()
#    ep = 0 
#    loss_hist = [softmax_loss(X, y, W, l)] # store history of loss 
#    N = X.shape[0]
#    nbatches = int(np.ceil(float(N)/batch_size))
#    while ep < nepoches: 
#        ep += 1 
#        mix_ids = np.random.permutation(N) # mix data 
#        for i in range(nbatches):
            # get the i-th batch
#            batch_ids = mix_ids[batch_size*i:min(batch_size*(i+1), N)] 
#            X_batch, y_batch = X[batch_ids], y[batch_ids]
#            W -=  lr * softmax_grad(X_batch, y_batch, W, l)
#        loss_hist.append(softmax_loss(X, y, W, l))
#        if np.linalg.norm(W - W_old)/W.size < tol:
#            break 
#        W_old = W.copy()
#    return W, loss_hist  

def softmax_fit(X, y, W,l,  lr = 1, iterations = 10):
    loss_hist = [softmax_loss(X, y, W, l)]
    for it in range(iterations):
        grad = softmax_grad(X, y, W, l)
        gti = grad**2
        adjusted_grad = grad / (1e-6 + np.sqrt(gti))
        W -=  lr * adjusted_grad
        loss_hist.append(softmax_loss(X, y, W, l))
    return W, loss_hist 

W_init = np.random.randn(flat_partial_train.shape[1], len(partial_label_train[1]))
W, loss_hist = softmax_fit(flat_partial_train, partial_label_train , W_init, iterations = 800, lr = 0.01, l = .00)
    
np.argmin(loss_hist)

plt.plot(loss_hist)
plt.xlabel('number of epoches', fontsize = 13)
plt.ylabel('loss', fontsize = 13)
plt.tick_params(axis='both', which='major', labelsize=13)
plt.show() 


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
