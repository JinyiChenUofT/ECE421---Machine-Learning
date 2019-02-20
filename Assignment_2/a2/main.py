import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from starter import loadData, softmax

if __name__ == '__main__':
    #trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
    #print (trainData.shape)
    #print (trainTarget.shape)

    test = np.array([[1,2],[2,4],[3,8]])
    print (test.shape)
    res = softmax(test)
    print (res.shape)
    #print (np.sum(test,axis=1))

