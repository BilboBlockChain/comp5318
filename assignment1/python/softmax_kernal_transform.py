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

#create val labels
label_val = label_train[25000:,]
partial_label_train = label_train[: 25000,]




#reshape data
def reshape_matrix(x):
   x_1=np.add.reduceat(x,np.arange(0,28,2),1)
   x_2=np.add.reduceat(x_1,np.arange(0,28,2),2)
   x_v=x_2.reshape(x_2.shape[0],-1)
   return x_v

flat_partial_train = reshape_matrix(partial_train)
flat_data_val = reshape_matrix(data_val)

flat_partial_train = partial_train
flat_data_val = data_val

#def reshape_svd(A, comps):
#    A = A.reshape(A.shape[0], A.shape[1] * A.shape[2])
#    
#   U, s, Vt = np.linalg.svd(A, full_matrices=False)
#    S = np.diag(s)
#    A_hat = U[0:U.shape[0], 0:comps].dot(S[0:comps,0:comps])
#    comp_ratio = (A.shape[1]*comps + comps + A.shape[0]*comps)/(A.shape[1] * A.shape[0])
#    print(comp_ratio)
#    return A_hat


#flat_partial_train = reshape_svd(partial_train,196)
#flat_data_val = reshape_svd(data_val,196)



#create softmax function 
def softmax(Z): # Todo 
    e_Z = np.exp(Z)
    A = e_Z / e_Z.sum(axis = 1, keepdims = True)
    return A

#define softmax gradient 
def softmax_grad(x, y, w):
    A = softmax(x.dot(w))    # shape of (N, C)
    id0 = range(x.shape[0])  # number of train data
    A[id0, y] -= 1           # A - Y, shape of (N, C)
    return x.T.dot(A)/x.shape[0]

    
#define softmax loss 
def softmax_loss(x, y, w):
    A = softmax(x.dot(w))
    id0 = range(x.shape[0])
    return -np.mean(np.log(A[id0, y]))


####other loop

def softmax_fit(X, y, W, lr = 0.01, nepoches = 20, tol = 1e-5, batch_size = 1000):
    W_old = W.copy()
    ep = 0 
    loss_hist = [softmax_loss(X, y, W)] # store history of loss 
    N = X.shape[0]
    nbatches = int(np.ceil(float(N)/batch_size))
    while ep < nepoches: 
        ep += 1 
        mix_ids = np.random.permutation(N) # mix data 
        for i in range(nbatches):
            # get the i-th batch
            batch_ids = mix_ids[batch_size*i:min(batch_size*(i+1), N)] 
            X_batch, y_batch = X[batch_ids], y[batch_ids]
            W -=  lr * softmax_grad(X_batch, y_batch, W)
        loss_hist.append(softmax_loss(X, y, W))
        if np.linalg.norm(W - W_old)/W.size < tol:
            break 
        W_old = W.copy()
    return W, loss_hist 

W_init = np.random.randn(flat_partial_train.shape[1], len(np.unique(partial_label_train)))
W, loss_hist = softmax_fit(flat_partial_train, partial_label_train , W_init, batch_size = 10, nepoches = 100, lr = 0.05)
    

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
print("Accurancy of model on test set:",accurancy(y_pre,label_val))