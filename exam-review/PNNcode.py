'''
Implementation of some algorithms in PNN,
a suppliment of KCL KNN lecture notes

@ Author: Ricky Zhu
@ email:  rickyzhu@foxmail.com
'''
from __future__ import print_function
import numpy as np
import pandas as pd


# ================================================
#             utility functions
# ================================================

def augmented_vectors(X, labels, normalised=False):
    '''
    X:      2d numpy array, samples data, where rows are 
            samples and columns are features
    labels: 1d numpy array of 1 or -1, labels of samples
    '''
    if normalised:
        aug_X = np.hstack([np.ones(len(X)).reshape([-1,1]), X])
        return np.array([labels[i] * aug_X[i] for i in range(len(aug_X))])
    else:
        return np.hstack([np.ones(len(X)).reshape([-1,1]), X])
    
def format_logdata(log_data):
    '''formatize log data using pandas.DataFrame'''
    df = pd.DataFrame(data=log_data[1:], columns=log_data[0])
    df.set_index(log_data[0][0])
    return df


# ======================================================
#                trainning procedures
# ======================================================


def sequential_perceptron_learning(Y, a, labels, eta=1, max_iter=10, print_log=False):
    '''
    sequential perceptron learning to train discriminant function

    Y:        2d array - training data in augmented notation
    a:        1d array - initial weights
    labels:   1d array - labels of corresponding sample in Y
    eta:      float - learning rate
    max_iter: int - max iterations to train
    print_log: bool - whether to print out log data
    '''
    log_data = [['iteration', 'a^t', 'y^t[k]','a_new', 'labels[k]', 'g(x)']]
    for i in range(max_iter):
        k = i % len(Y)
        if (a.dot(Y[k]) * labels[k]) < 0:  # misclassifed
            a_new = a + eta * labels[k] * Y[k]
        else:
            a_new  = a
        log_data.append([i+1, a, Y[k], a_new, labels[k], a.dot(Y[k])])
        a = a_new
    if print_log:
        print(format_logdata(log_data))
    return a

def sequential_LMS_learning(Y, a, b, alpha=1, max_iter=10, print_log=False):
    '''
    Sequential Widrow-Hoff (LMS) Learning Algorithm to train
    discriminant function
    
    Y:        2d array - training data in augmented notation
    a:        1d array - initial weights
    b:        1d array - positive-valued margin vector
    eta:      float - learning rate
    max_iter: int - max iterations to train
    print_log: bool - whether to print out log data
    '''
    log_data = [['iteration', 'a^t', 'y^t_k', 'a_new', 'g(x)=ay']]
    for i in range(max_iter):
        k = i % len(Y)
        a_new = a + alpha * (b[k] - a.dot(Y[k])) * Y[k]
        log_data.append([i+1, a, Y[k], a_new, a.dot(Y[k])])
        a = a_new
    if print_log:
        print(format_logdata(log_data))
    return a

def batch_perceptron_learning(Y, a, labels, learning_rate=0.1, max_iter=10):
    log_data = [['iteration', 'a^t', 'a_new' ]]
    for i in range(max_iter):
        accumulated = 0
        for k in range(len(Y)):
            if (a.dot(Y[k]) * labels[k]) < 0:  # misclassifed
                accumulated += learning_rate * labels[k] * Y[k]
        a_new = a + accumulated
        log_data.append([i+1, a, a_new])
        a = a_new
    return log_data

def batch_LMS_learning(Y, a, b, learning_rate=0.1, max_iter=10):
    '''
    Widrow-Hoff (LMS) method
    '''
    log_data = [['iteration', 'a^t', 'a_new']]
    for i in range(max_iter):
        a_new = a - learning_rate * Y.T.dot(Y.dot(a) - b)
        log_data.append([i+1, a, a_new])
        a = a_new
    return log_data

# X = np.array([
#     [0,0],
#     [1,0],
#     [2,1],
#     [0,1],
#     [1,2]
# ])
# labels = np.array([1,1,1,-1,-1])
# a = np.array([-1.5, 5, -1])
# Y1= augmented_vectors(X, labels)
# print(format_logdata(batch_perceptron_learning(Y1, a, labels, 1 )))

# b = np.array([2,2,2,2,2])
# Y2 = augmented_vectors(X, labels, normalised=True)
# print(format_logdata(batch_LMS_learning(Y2, a, b, 0.1, 1000)))

def sequential_delta_learning(X, w, labels, eta=0.1, max_iter=10):
    '''
    sequential delta learning for Linear Threshold Unit

    X:        2d array - training data in augmented notation
    w:        1d array - initial weights
    labels:   1d array - labels of corresponding sample in Y
    eta:      float - learning rate
    max_iter: int - max iterations to train
    print_log: bool - whether to print out log data
    '''
    log_data = [['iteration', 'label[k]', 'X[k]', 'w', 'H(wx)', 'eta*(label[k]-H(wx))', 'w_new']]
    for i in range(max_iter):
        def H(wx):
            if wx > 0: return 1
            else: return 0
        k = i % len(X)
        w_new = w + eta * (labels[k] - H(w.dot(X[k]))) * X[k]
        log_data.append([i+1, labels[k], X[k], w, H(w.dot(X[k])), eta*(labels[k]-H(w.dot(X[k]))), w_new])
        w = w_new
    if print_log:
        print(format_logdata(log_data))
    return w

def sequential_hebbian_learning(x, W, alpha=0.1, max_iter=10):
    '''sequential Hebbian learning for negative feedback network'''
    log_data = [['iteration', 'e', 'We', 'y', 'Wy']]
    e = x
    y = np.zeros(W.shape[0])
    for i in range(max_iter):
        y = y + alpha*W.dot(e)
        log_data.append([i+1, e, W.dot(e), y, W.T.dot(y)])
        e = x - W.T.dot(y).T
    return log_data

# X = np.array([
#     [0,2],
#     [1,2],
#     [2,1],
#     [-3,1],
#     [-2,-1],
#     [-3,-2]
# ])
# labels = np.array([1,1,1,0,0,0])
# w = np.array([1, 0, 0])
# aug_X = augmented_vectors(X, labels)
# print(format_logdata(sequential_delta_learning(aug_X, w, labels, 1, 13)))

# W = np.array([[1,1,0], [1,1,1]])
# x = np.array([1,1,0])
# print(format_logdata(sequential_hebbian_learning(x, W, 0.25, 5)))


# =========================================================
#                   feature extraction
# =========================================================

def PCA_KLT(X, k, print_log=False):
    '''
    PCA for X using Karhunen-Loeve Transform. Return the top k components.
    
    X: 2darray - r by c matrix where row represent sample and column represent feature
    k: int - how many components to return 
    print_log: bool - if print out all intermediate results
    '''
    mean_vector = np.mean(X, axis=0)
    Xp = X - mean_vector
    def covM(_X):
        M = np.zeros([_X.shape[1], _X.shape[1]])
        for i in range(len(_X)):
            M += _X[i].reshape([-1,1]).dot(_X[i].reshape([1,-1]))
        return M / len(_X)
    X_cov = covM(Xp)
    E, V = np.linalg.eig(X_cov)
    print([V.T[np.argsort(-E)[i]] for i in range(k)])
    projector = np.array([V.T[np.argsort(-E)[i]] for i in range(k)])
    if print_log:
        print('mean vector:', mean_vector)
        print('non-zero vectors:', Xp)
        print('Cov Mat:', X_cov)
        print('projector:', projector)
        print('Cov Mat of transformd data:',np.round(covM(projector.dot(Xp.T).T),5))
    return projector.dot(Xp.T)

# x1 = np.array([1,2,1])
# x2 = np.array([2,3,1])
# x3 = np.array([3,5,1])
# x4 = np.array([2,2,1])
# X = np.array([x1,x2,x3,x4])
# PCA_KLT(X, k=2, print_log=True)

def batch_oja_learning(X, w, eta=0.01, epoch=2, print_log=False):
    '''
    Oja batch learning for neural network PCA

    X:        2d array - r by c matrix where row represent 
                  sample and column represent feature
    w:        1d array - initial weights
    eta:      float - learning rate
    epoch:    int - max epoch to train
    print_log: bool - whether to print out intermediate results
    '''
    mean_vector = np.mean(X, axis=0)
    Xp = X - mean_vector
    logdata = [['epoch', 'Xp[t]', 'y=wx', 'X[t]-yw', 'eta* y * (X[t]-yw)']]
    for i in range(epoch):
        delta_w = 0
        
        for t in range(len(Xp)):
            intermediates = [i+1]
            y = w.dot(Xp[t])
            intermediates.append(Xp[t])
            intermediates.append(y)
            intermediates.append(Xp[t].T - y*w)
            intermediates.append(eta*y*(Xp[t].T - y*w))
            delta_w += eta*y*(Xp[t].T - y*w)
            logdata.append(intermediates)
        w = w + delta_w
    if print_log:
        print(format_logdata(logdata))
    return w

# X = np.array([
#     [0,1],
#     [3,5],
#     [5,4],
#     [5,6],
#     [8,7],
#     [9,7]
# ])
# w=np.array([-1, 0])
# batch_oja_learning(X, w, 0.01, 2, True)

def LDA_J(w, X, labels, print_log=False):
    '''
    compute the cost of Fisher's LDA, only for binary classification

    w:         1d array - weights that project sample to a line
    X:         2d array - sample data, row as sample and col as feature
    labels:    1d array - labels of corresponding sample
    print_log: bool - whether to print out intermediate results
    '''
    classes = list(set(labels))
    M = np.array([np.mean(X[labels==y], axis=0) for y in classes])
    sb = w.dot(M[0] - M[1]) ** 2
    sw = np.sum([w.dot(x-M[0])**2 for x in X[labels==classes[0]]]) \
        + np.sum([w.dot(x-M[1])**2 for x in X[labels==classes[1]]])
    if print_log:
        print('sb:', sb, ', sw:', sw)
    return sb / sw

# X = np.array([
#     [1,2],
#     [2,1],
#     [3,3],
#     [6,5],
#     [7,8]
# ])
# labels = np.array([1,1,1,2,2])
# w1 = np.array([-1, 5])
# w2 = np.array([2, -3])

# print(LDA_J(w1, X, labels, True))
# print(LDA_J(w2, X, labels, True))

def extreme_learning_machine(X, V, w, func_g=None, print_log=False):
    '''
    random projection using extreme learning machine

    X:        2d array - sample data, COLOUM as sample, and ROW as feature
    V:        2d array - weights that project x to y
    w:        1d array - weights that map y to t(targets of sample)
    func_g:   function - that g(Vx) that map x to y
    print_log: bool - whether to print out intermediate results
    ''' 
    if func_g == None:
	def func_g(X, V):
            return np.where(V.dot(X)<0, 0, 1)
    Y = func_g(X, V)
    if print_log:
        print('Y (output of hidden layer): ')
        print(Y)
    return w.dot(np.vstack([np.ones(Y.shape[1]),Y]))

# V = np.array([
#     [-0.62, 0.44, -0.91],
#     [-0.81, -0.09, 0.02],
#     [0.74, -0.91, -0.60],
#     [-0.82, -0.92, 0.71],
#     [-0.26, 0.68, 0.15],
#     [0.80, -0.94, -0.83],
# ])
# X = np.array([
#     [1,1,1,1],
#     [0,0,1,1],
#     [0,1,0,1]
# ])
# w = np.array([0,0,0,-1,0,0,2])
# print(extreme_learning_machine(X,V,w,print_log=True))

def spaCodRecErr(x, V, y):
    '''
    reconstruction error for dictionary-based sparse coding
    
    x: a dense vector
    V: the dictionary
    y: the transfered sparse coding of x
    '''
    return np.linalg.norm(x - V.dot(y))


# ========================================================
#               support vector machine
# ========================================================

def compute_svm_weights(X_supp, Y_supp, print_log=False):
    '''
    compute weights of a linear SVM with identified support vectors

    X_supp: 2d array - each row represents a support vector
    Y_supp: 1d array - label of each support vector, either 1 or -1
    '''
    A = X_supp.dot(X_supp.T)*Y_supp
    A = np.vstack([A, Y_supp])
    A = np.hstack([A, np.array([1]*len(X_supp)+[0]).reshape([-1,1])])
    b = np.concatenate([Y_supp, [0]])
    res = np.linalg.inv(A).dot(b)
    lambdas, w0 = res[:-1], res[-1]
    w = np.sum([lambdas[i]*X_supp[i]*Y_supp[i] for i in range(len(X_supp))], axis=0)
    if print_log:
        print('euqation atrix:', A)
        print('equation vector:', b)
        print('lambdas:',lambdas, ', w0:', w0)
    return (w, w0)

# X_supp = np.array([
#      [3,1],
#      [3,-1],
#      [1,0]
#      ]
# )
# Y_supp = np.array([1,1,-1])
# print(compute_svm_weights(X_supp, Y_supp, print_log=True))

# ==============================================================
#                       clustering
# ==============================================================



# =============================================================
#                      tree and forest
# =============================================================

