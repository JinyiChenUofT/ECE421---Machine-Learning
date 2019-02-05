import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from starter.py import loadData, grad_descent


if __name__ == '__main__':
    '''
    #initialize session
    init = tf.global_variables_initializer()

    sess = tf.InteractiveSession()
    sees.run(init)
    

    #training the model
    
    #for step in range(0,200):
        _, err, currentW, currentb, yhat = sess.run()
    '''
    
    trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
    #print (trainData.shape) (3500, 28, 28) 28*28 = 784
    #print (validData.shape) (100, 28, 28)
    #print (testData.shape) (145, 28, 28)
    
    #print (trainTarget.shape)
    #print (trainTarget.shape[1])
    #print (testData.shape)
    #print (validData.shape)
    train_X = trainData.reshape(3500,784)
    val_X = validData.reshape(100,784)
    test_X = testData.reshape(145,784)

    W = np.zeros((1, 784))
    b = 0
    '''
    loss_set = [1,2,3,4,7,4,2,1]

    plt.figure(1)
    plt.plot(loss_set)
    #plt.plot(np.arange(len(loss_set)),loss_set)
    plt.savefig("loss.png")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.show()
    '''
    grad_descent(W,b,train_X,trainTarget,val_X,validTarget,test_X,testTarget, alpha=0.005,epochs=5000,reg=0,error_tol=0.0000001)
    #new_W, new_b = grad_descent(W,b,train_X,y=trainTarget,alpha=0.005,epochs=5000,reg=0,error_tol=0.0000001)
    #mse()
    #return