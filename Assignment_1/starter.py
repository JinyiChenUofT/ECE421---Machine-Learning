import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def loadData():
    with np.load('notMNIST.npz') as data :
        Data, Target = data ['images'], data['labels']
        posClass = 2
        negClass = 9
        dataIndx = (Target==posClass) + (Target==negClass)
        Data = Data[dataIndx]/255.
        Target = Target[dataIndx].reshape(-1, 1)
        Target[Target==posClass] = 1
        Target[Target==negClass] = 0
        np.random.seed(421)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data, Target = Data[randIndx], Target[randIndx]
        trainData, trainTarget = Data[:3500], Target[:3500]
        validData, validTarget = Data[3500:3600], Target[3500:3600]
        testData, testTarget = Data[3600:], Target[3600:]
    return trainData, validData, testData, trainTarget, validTarget, testTarget

#MSE is the sum of squared distances between our target variable and predicted values.
def MSE(W, b, x, y, reg):
    #f(w) = 1/2N * ||xw-y||^2 + reg/2 *||w||^2
    #y_predicted definition
    y_predicted = tf.matmul(x,W)+b

    #error definition
    N = y.shape[1]
    mean_uqared_error = tf.reduce_sum((1/(2*N))*tf.square(y_predicted - y)+(reg/2)*tf.square(W), name = 'mean_uqared_error')
    return mean_uqared_error


def gradMSE(W, b, x, y, reg):
    W_tranpose = np.transpose(W)
    
    x_hat = tf.matmul(tf.matmul(np.linalg.inv(tf.matmul(W_tranpose,W)),W_tranpose),b)

    return x_hat

#def crossEntropyLoss(W, b, x, y, reg):
    # Your implementation here

#def gradCE(W, b, x, y, reg):
    # Your implementation here

#def grad_descent(W, b, trainingData, trainingLabels, alpha, iterations, reg, EPS):
    # Your implementation here

#def buildGraph(beta1=None, beta2=None, epsilon=None, lossType=None, learning_rate=None):
    # Your implementation here

if __name__ == '__main__':
    #initialize session
    init = tf.global_variables_initializer()

    sess = tf.InteractiveSession()
    sees.run(init)

    #training the model
    
    #for step in range(0,200):
        _, err, currentW, currentb, yhat = sess.run()
    #trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
    #mse()
    #return