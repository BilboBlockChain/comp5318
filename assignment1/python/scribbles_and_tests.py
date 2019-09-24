import numpy as np

C = 5    # number of classes
N = 800  # number of points per class 
means = [[2, 2], [8, 3], [3, 6], [14, 2], [12, 8]]
cov = [[1, 0], [0, 1]]

X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)
X2 = np.random.multivariate_normal(means[2], cov, N)
X3 = np.random.multivariate_normal(means[3], cov, N)
X4 = np.random.multivariate_normal(means[4], cov, N)

X = np.concatenate((X0, X1, X2, X3, X4), axis = 0) # each row is a datapoint
Xbar = np.concatenate((X, np.ones((X.shape[0], 1))), axis = 1) # bias trick 
y = np.asarray([0]*N + [1]*N + [2]*N+ [3]*N + [4]*N)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(Xbar, y, test_size=0.33, random_state=42)


W_init = np.random.randn(X_train.shape[1], C)

def softmax(X):
    return np.exp(X)/ np.sum(np.exp(X), axis = 0)



# cost or loss function  
def softmax_loss(X, y, W):
    A = softmax(X.dot(W))
    id0 = range(X.shape[0])
    return -np.mean(np.log(A[id0, y]))


softmax_loss(X_train, y_train, W_init)



A = softmax(X_train.dot(W_init))
id0 = range(X_train.shape[0])

A[id0,]

y_train.shape

X_train.shape

W_init.shape

A[id0, y_train].shape
A.shape
