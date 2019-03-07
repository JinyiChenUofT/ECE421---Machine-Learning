import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import math
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# Load the data
def loadData():
    with np.load("notMNIST.npz") as data:
        Data, Target = data["images"], data["labels"]
        np.random.seed(521)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data = Data[randIndx] / 255.0
        Target = Target[randIndx]
        trainData, trainTarget = Data[:10000], Target[:10000]
        validData, validTarget = Data[10000:16000], Target[10000:16000]
        testData, testTarget = Data[16000:], Target[16000:]
    return trainData, validData, testData, trainTarget, validTarget, testTarget

# Implementation of a neural network using only Numpy - trained using gradient descent with momentum


def convertOneHot(trainTarget, validTarget, testTarget):
    newtrain = np.zeros((trainTarget.shape[0], 10))
    newvalid = np.zeros((validTarget.shape[0], 10))
    newtest = np.zeros((testTarget.shape[0], 10))

    for item in range(0, trainTarget.shape[0]):
        newtrain[item][trainTarget[item]] = 1
    for item in range(0, validTarget.shape[0]):
        newvalid[item][validTarget[item]] = 1
    for item in range(0, testTarget.shape[0]):
        newtest[item][testTarget[item]] = 1
    return newtrain, newvalid, newtest


def shuffle(trainData, trainTarget):
    np.random.seed(421)
    randIndx = np.arange(len(trainData))
    target = trainTarget
    np.random.shuffle(randIndx)
    data, target = trainData[randIndx], target[randIndx]
    return data, target


def relu(x):
    # TODO
    return np.maximum(x, 0)


def softmax(x):
    # TODO
    #return np.exp(x)/np.sum(np.exp(x))
    return np.exp(x)/np.sum(np.exp(x),axis=1,keepdims=True)


def computeLayer(X, W, b):
    # TODO
    return np.matmul(X, W)+b


def CE(target, prediction):
    # TODO
    score = softmax(prediction)
    ce = np.sum(np.multiply(target, np.log(score)), axis=1)
    loss = -np.mean(ce)
    return loss


def gradCE(target, prediction):
    # TODO
    N = target.shape[0]
    score = softmax(prediction)
    print ("score.shape: ", score.shape)
    print ("target.shape: ",target.shape)
    res = score - prediction
    return res


def error(target, prediction):
    return np.multiply((target-prediction),(target-prediction))


def derivation_LW(last_X, y, target):
    grad_CE = gradCE(target, y)
    res = 2*np.multiply((np.multiply((y-target), grad_CE)), last_X)
    return res
    
def backPropagation(xi,sh,xh,so,wo,prediction,target):
    #xi 10000, 784
    #sh, xh  10000, 10000
    #so 10000, 10
    #wo 10000, 10
    #target, prediciton delta_o 10000,10
    grad_ce = gradCE(target,prediction) #delta_o 10000,10
    
    der_wo = np.dot(xh,grad_ce)
    der_bo = np.mean(1,grad_ce) #? #der_bo = np.dot(1,gred_ce)

    der_e_xiw = np.dot(grad_ce,np.transpose(wo))
    der_e_xiw = np.where(sh>0,der_e_xiw,0)  #delta_h 10000, 10000
    der_wh = np.dot(np.transpose(xi),der_e_xiw) 

    #der_e_xib = np.dot(grad_ce,transpose)
    #der_e_xib = np.where(sh>0,der_e_xib,0) #delta_h
    #der_bh = np.dot(1,der_e_xib)
    der_bh = np.mean(1,der_e_xiw) #? #der_bh = np.mean(der_e_xiw,axis=0)

    print ('der_bo.shape: ',der_bo)
    print ('der_bh.shape: ',der_bh)
    return der_wo, der_bo, der_wh, der_bh

def classify_result(Data, Target, W, b):
    N = Target.shape[0]
    classifier = np.zeros((N, 1))
    predict = np.dot(Data, W.T)
    loss = 0

    for i in range(N):
        if predict[i] > b:
            classifier[i] = 1
        else:
            classifier[i] = 0

        if Target[i] != classifier[i]:
            loss = loss+1

    print(classifier.shape)
    print(Target.shape)
    return classifier, loss


def train_network(epochs=200):
    trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
    train_y, valid_y, test_y = convertOneHot(trainTarget, validTarget, testTarget)

    accuracy_set = []
    print ("trainTarget.shape: ",trainTarget.shape)
    print ("newtrain.shape: ", train_y.shape) #10000, 10
    s, l, h = trainData.shape #10000 samples, 28, 28
    F = l*h  #784 features
    c = 10  #10 classes
    xi = trainData.reshape(s, F) #10000, 784
    variance_h = 2/(F + s)
    variance_o = 2/(s + c)

    mean, stand_dev_h, stand_dev_o = 0, math.sqrt(variance_h), math.sqrt(variance_o), 

    Wh = np.random.normal(mean, stand_dev_h, (F, s)) #784,10000
    bh = np.zeros((1,s))                           #1, 10000
    Wo = np.random.normal(mean, stand_dev_o, (s, c)) #10000, 10
    bo = np.zeros((1,c))                           #1, 10


    v_Wh = np.full((F, s), 1e-5)
    v_bh = np.full((1, s), 1e-5)
    v_Wo = np.full((s, c), 1e-5)
    v_bo = np.full((1,c), 1e-5)

    gamma = 0.99
    learning_rate = 0.01

    i = 0
    while i < epochs:
        # forward propagate
        if i%10 == 0:
            print ("iteration ",i)
        sh = computeLayer(xi, Wh, bh)
        xh = relu(sh)    #10000, 10000
        #print ("xh's shape: ", xh.shape)
        so = computeLayer(xh, Wo, bo)
        yo = softmax(so)  #10000, 10
        #print ("yo's shape: ", yo.shape)
        
        pred = np.argmax(yo, axis = 1)
        target = np.argmax(train_y, axis = 1)
        
        acc=np.mean(pred == target)
        accuracy_set.append(acc)

        # backward propagate
        der_wo, der_bo, der_wh, der_bh = backPropagation(xi,sh,xh,so,Wo,yo,train_y)

        v_Wh = gamma*v_Wh + learning_rate*der_wh
        v_bh = gamma*v_bh + learning_rate*der_bh
        v_Wo = gamma*v_Wo + learning_rate*der_wo
        v_bo = gamma*v_bo + learning_rate*der_bo

        Wh = Wh - v_Wh
        bh = bh - v_bh
        Wo = Wo - v_Wo
        bo = bo - v_bo

    print(Wo)
    a, b = classify_result(trainData, trainTarget, Wo, bo)
    print(a)
    print(b)
    
    plt.figure(1)
    plt.title("CE loss set: ephocs=200 learning rate=0.0001")
    plt.plot(accuracy_set)
    pic_name = "accuracy_set.png"
    plt.xlabel("epochs")
    plt.ylabel("accuracy_set")
    plt.show()
    return

