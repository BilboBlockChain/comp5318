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

#create val labels
label_val = label_train[20000:,]
partial_label_train = label_train[: 20000,]

#data matrix already contained between 0 and 1 no transformation needed
np.min(flat_train)
np.max(flat_train)

#one hot encode y for softmax output 
def oneHot(y):
    zeroesY = np.zeros((y.size, y.max() + 1))
    zeroesY[np.arange(y.size), y] = 1
    return zeroesY


#create softmax function 
def softmax(x):
    return np.exp(x)/ np.sum(np.exp(x), axis = 0)

#retrieve probabilites and arg max predictions
def outputs(X):
    probs = softmax(np.dot(X,w))
    preds = np.argmax(probs, axis = 1)
    return probs, preds


def getLoss(w,x,y,lam):
    m = x.shape[0] #First we get the number of training examples
    y_mat = oneHot(y) #Next we convert the integer class coding into a one-hot representation
    scores = np.dot(x,w) #Then we compute raw class scores given our input and current weights
    prob = softmax(scores) #Next we perform a softmax on these scores to get their probabilities
    loss = (-1 / m) * np.sum(y_mat * np.log(prob)) + (lam/2)*np.sum(w*w) #We then find the loss of the probabilities
    grad = (-1 / m) * np.dot(x.T,(y_mat - prob)) + lam*w #And compute the gradient for that loss
    return loss,grad
    

w = np.zeros([flat_partial_train.shape[1],len(np.unique(partial_label_train))])
lam = 1
iterations = 100
learningRate = 1e-3
losses = []
for i in range(0,iterations):
    loss,grad = getLoss(w,flat_partial_train, partial_label_train,lam)
    losses.append(loss)
    w = w - (learningRate * grad)
print(loss)

plt.plot(losses)
