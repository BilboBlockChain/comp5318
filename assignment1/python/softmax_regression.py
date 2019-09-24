import h5py
import numpy as np
import os
import scipy as sp

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




#create val set
data_val = data_train[25000:,]
partial_train = data_train[:25000,]

#create val labels
label_val = label_train[25000:,]
partial_label_train = label_train[: 25000,]



#reshape data and svd 
def reshape_svd(A, comps):
    A = A.reshape(A.shape[0], A.shape[1] * A.shape[2])
    
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    S = np.diag(s)
    A_hat = U[0:U.shape[0], 0:comps].dot(S[0:comps,0:comps])
    comp_ratio = (A.shape[1]*comps + comps + A.shape[0]*comps)/(A.shape[1] * A.shape[0])
    print(comp_ratio)
    return A_hat

flat_partial_train = reshape_svd(partial_train, 50)


flat_partial_train.shape

#data matrix already contained between 0 and 1 no transformation needed
np.min(flat_partial_train)
np.max(flat_partial_train)

#one hot encode y for softmax output 
def oneHot(y):
    zeroesY = np.zeros((y.size, y.max() + 1))
    zeroesY[np.arange(y.size), y] = 1
    return zeroesY


oneHot(partial_label_train)


np.unique(partial_label_train)

#create softmax function 
def softmax(Z): # Todo 
    e_Z = np.exp(Z)
    A = e_Z / e_Z.sum(axis = 1, keepdims = True)
    return A

#retrieve probabilites and arg max predictions
def outputs(X):
    probs = softmax(np.dot(X,w))
    preds = np.argmax(probs, axis = 1)
    return probs, preds

#define loss functiona and gradient operation 
def getLoss(w,x,y,lam):
    m = x.shape[0] #First we get the number of training examples
    y_vec = oneHot(y) #Next we convert the integer class coding into a one-hot representation
    raw = np.dot(x,w) #Then we compute raw class scores given our input and current weights
    prob = softmax(raw) #Next we perform a softmax on these scores to get their probabilities
    loss = (-1 / m) * np.sum(y_vec * np.log(prob)) + (lam/2)*np.sum(w*w) #We then find the loss of the probabilities
    grad = (-1 / m) * np.dot(x.T,(y_vec - prob)) + lam*w #And compute the gradient for that loss
    return loss,grad
    
#create sgd loop over loss function and grad
w = np.zeros([flat_partial_train.shape[1],len(np.unique(partial_label_train))])
lam = 1
iterations = 10000
learningRate = 1e-3
losses = []
for i in range(0,iterations):
    loss,grad = getLoss(w,flat_partial_train, partial_label_train,lam)
    losses.append(loss)
    w = w - (learningRate * grad)
print(loss)

plt.plot(losses)
