import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import json
from enum import Enum
import time

class ModelType(Enum):
    TrainModel = 1
    ValidationModel = 2
    TestModel = 3


def loadData():
    with np.load('notMNIST.npz') as data:
        Data, Target = data['images'], data['labels']
        posClass = 2
        negClass = 9
        dataIndx = (Target == posClass) + (Target == negClass)
        Data = Data[dataIndx]/255.
        Target = Target[dataIndx].reshape(-1, 1)
        Target[Target == posClass] = 1
        Target[Target == negClass] = 0
        np.random.seed(421)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data, Target = Data[randIndx], Target[randIndx]
        trainData, trainTarget = Data[:3500], Target[:3500]
        validData, validTarget = Data[3500:3600], Target[3500:3600]
        testData, testTarget = Data[3600:], Target[3600:]
    return trainData, validData, testData, trainTarget, validTarget, testTarget

# MSE is the sum of squared distances between our target variable and predicted values.


def MSE(W, b, x, y, reg):
    # f(w) = 1/2N * ||xw-y||^2 + reg/2 *||w||^2
    # y_predicted definition
    W_tranpose = np.transpose(W)
    y_predicted = np.matmul(x, W_tranpose)+b
    # loss term
    N = y.shape[0]
    loss_term = (1/(2*N))*np.sum(np.square(y_predicted-y))

    # regularization term
    reg_term = (reg/2)*np.sum(np.matmul(W, W_tranpose))

    # loss function definition
    mean_suqared_error = loss_term+reg_term
    return mean_suqared_error


def gradMSE(W, b, x, y, reg):
    N = y.shape[0]
    W_tranpose = np.transpose(W)
    y_predicted = np.matmul(x, W_tranpose)+b

    # gradient with respect to weight
    grad_w_loss = (1/N)*np.dot(np.transpose(x), (y_predicted-y))
    grad_w_loss = np.transpose(grad_w_loss)
    grad_w_reg = reg*W
    grad_w = grad_w_loss + grad_w_reg

    # gradient with respect to bias
    grad_b = (1/N)*np.sum(y_predicted-y)

    return grad_w, grad_b


'''
def grad_descent(W, b, x,y, alpha, epochs, reg, error_tol):
    i = 0
    w_step_size = 0
    train_loss_set = []
    while i < epochs:

        grad_w, grad_b  = gradMSE(W,b,x,y,reg)
        w_step_size = alpha * grad_w
        b_step_size = alpha * grad_b
        W = W - w_step_size
        b = b - b_step_size
        train_mse = MSE(W,b,x,y,reg)
        train_loss_set.append(train_mse)
        print("iteration: ",i)
        print (np.linalg.norm(w_step_size))
        if (np.linalg.norm(w_step_size) <= error_tol):
            break
       
        i = i + 1
    print ("Trainning model\n")
    plt.figure(1)
    plt.title("trainning loss set")
    plt.plot(train_loss_set)
    plt.xlabel("epochs")
    plt.ylabel("loss")
    #plt.legend()
    plt.show()
'''
# def grad_descent(W, b, x, y, alpha, epochs, reg, error_tol, model):


def grad_descent(W, b, train_x, train_y, val_x, val_y, test_x, test_y, alpha, epochs, reg, error_tol):
    i = 0
    W1 = W2 = W3 = W
    b1 = b2 = b3 = b
    train_loss_set1 = []
    train_loss_set2 = []
    train_loss_set3 = []
    val_loss_set1 = []
    val_loss_set2 = []
    val_loss_set3 = []
    test_loss_set1 = []
    test_loss_set2 = []
    test_loss_set3 = []
    w_step_size = 0
    alpha2 = 0.001
    alpha3 = 0.0001

    # while i < epochs and abs(np.linalg.norm(w_step_size))<=error_tol:
    while i < epochs:
        grad_w1, grad_b1 = gradMSE(W1, b1, train_x, train_y, reg)
        grad_w2, grad_b2 = gradMSE(W2, b2, val_x, val_y, reg)
        grad_w3, grad_b3 = gradMSE(W3, b3, test_x, test_y, reg)

        w_step_size1 = alpha * grad_w1
        b_step_size1 = alpha * grad_b1
        W1 = W1 - w_step_size1
        b1 = b1 - b_step_size1

        w_step_size2 = alpha2 * grad_w2
        b_step_size2 = alpha2 * grad_b2
        W2 = W2 - w_step_size2
        b2 = b2 - b_step_size2

        w_step_size3 = alpha3 * grad_w3
        b_step_size3 = alpha3 * grad_b3
        W3 = W3 - w_step_size3
        b3 = b3 - b_step_size3

        train_mse1 = MSE(W1, b1, train_x, train_y, reg)
        val_mse1 = MSE(W1, b1, val_x, val_y, reg)
        test_mse1 = MSE(W1, b1, test_x, test_y, reg)
        train_loss_set1.append(train_mse1)
        val_loss_set1.append(val_mse1)
        test_loss_set1.append(test_mse1)

        train_mse2 = MSE(W2, b2, val_x, val_y, reg)
        val_mse2 = MSE(W2, b2, val_x, val_y, reg)
        test_mse2 = MSE(W2, b2, test_x, test_y, reg)
        train_loss_set2.append(train_mse2)
        val_loss_set2.append(val_mse2)
        test_loss_set2.append(test_mse2)

        train_mse3 = MSE(W3, b3, train_x, train_y, reg)
        val_mse3 = MSE(W3, b3, val_x, val_y, reg)
        test_mse3 = MSE(W3, b3, test_x, test_y, reg)
        train_loss_set3.append(train_mse3)
        val_loss_set3.append(val_mse3)
        test_loss_set3.append(test_mse3)

        i = i + 1

    print("Trainning model\n")
    plt.figure(1)
    plt.title("trainning loss set: ephocs=5000 reg=0")
    plt.plot(train_loss_set1)
    plt.plot(train_loss_set2)
    plt.plot(train_loss_set3)
    pic_name = "training_loss.png"
    plt.savefig(pic_name)
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend(['alpha=0.005', 'alpha=0.001',
                'alpha=0.0001'], loc='upper right')

    print("Validation model")
    plt.figure(2)
    plt.title("Validation loss set: ephocs=5000 reg=0")
    plt.plot(val_loss_set1)
    plt.plot(val_loss_set2)
    plt.plot(val_loss_set3)
    pic_name = "validation_loss.png"
    plt.savefig(pic_name)
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend(['alpha=0.005', 'alpha=0.001',
                'alpha=0.0001'], loc='upper right')

    print("Test model")
    plt.figure(3)
    plt.title("Test loss set: ephocs=5000 reg=0")
    plt.plot(test_loss_set1)
    plt.plot(test_loss_set2)
    plt.plot(test_loss_set3)
    pic_name = "test_loss.png"
    plt.savefig(pic_name)
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend(['alpha=0.005', 'alpha=0.001',
                'alpha=0.0001'], loc='upper right')
    plt.show()



def grad_descent(W, b, train_x, train_y, val_x, val_y, test_x, test_y, alpha, epochs, reg, error_tol):
    i = 0
    W1 = W2 = W3 = W
    b1 = b2 = b3 = b
    train_loss_set1 = []
    train_loss_set2 = []
    train_loss_set3 = []
    val_loss_set1 = []
    val_loss_set2 = []
    val_loss_set3 = []
    test_loss_set1 = []
    test_loss_set2 = []
    test_loss_set3 = []
    w_step_size = 0
    alpha2 = 0.001
    alpha3 = 0.0001

    # while i < epochs and abs(np.linalg.norm(w_step_size))<=error_tol:
    while i < epochs:
        grad_w1, grad_b1 = gradMSE(W1, b1, train_x, train_y, reg)
        grad_w2, grad_b2 = gradMSE(W2, b2, val_x, val_y, reg)
        grad_w3, grad_b3 = gradMSE(W3, b3, test_x, test_y, reg)

        w_step_size1 = alpha * grad_w1
        b_step_size1 = alpha * grad_b1
        W1 = W1 - w_step_size1
        b1 = b1 - b_step_size1

        w_step_size2 = alpha2 * grad_w2
        b_step_size2 = alpha2 * grad_b2
        W2 = W2 - w_step_size2
        b2 = b2 - b_step_size2

        w_step_size3 = alpha3 * grad_w3
        b_step_size3 = alpha3 * grad_b3
        W3 = W3 - w_step_size3
        b3 = b3 - b_step_size3

        train_mse1 = MSE(W1, b1, train_x, train_y, reg)
        val_mse1 = MSE(W1, b1, val_x, val_y, reg)
        test_mse1 = MSE(W1, b1, test_x, test_y, reg)
        train_loss_set1.append(train_mse1)
        val_loss_set1.append(val_mse1)
        test_loss_set1.append(test_mse1)

        train_mse2 = MSE(W2, b2, val_x, val_y, reg)
        val_mse2 = MSE(W2, b2, val_x, val_y, reg)
        test_mse2 = MSE(W2, b2, test_x, test_y, reg)
        train_loss_set2.append(train_mse2)
        val_loss_set2.append(val_mse2)
        test_loss_set2.append(test_mse2)

        train_mse3 = MSE(W3, b3, train_x, train_y, reg)
        val_mse3 = MSE(W3, b3, val_x, val_y, reg)
        test_mse3 = MSE(W3, b3, test_x, test_y, reg)
        train_loss_set3.append(train_mse3)
        val_loss_set3.append(val_mse3)
        test_loss_set3.append(test_mse3)

        i = i + 1

    print("Trainning model\n")
    plt.figure(1)
    plt.title("trainning loss set: ephocs=5000 reg=0")
    plt.plot(train_loss_set1)
    plt.plot(train_loss_set2)
    plt.plot(train_loss_set3)
    pic_name = "training_loss.png"
    plt.savefig(pic_name)
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend(['alpha=0.005', 'alpha=0.001',
                'alpha=0.0001'], loc='upper right')

    print("Validation model")
    plt.figure(2)
    plt.title("Validation loss set: ephocs=5000 reg=0")
    plt.plot(val_loss_set1)
    plt.plot(val_loss_set2)
    plt.plot(val_loss_set3)
    pic_name = "validation_loss.png"
    plt.savefig(pic_name)
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend(['alpha=0.005', 'alpha=0.001',
                'alpha=0.0001'], loc='upper right')

    print("Test model")
    plt.figure(3)
    plt.title("Test loss set: ephocs=5000 reg=0")
    plt.plot(test_loss_set1)
    plt.plot(test_loss_set2)
    plt.plot(test_loss_set3)
    pic_name = "test_loss.png"
    plt.savefig(pic_name)
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend(['alpha=0.005', 'alpha=0.001',
                'alpha=0.0001'], loc='upper right')
    plt.show()
    return W, b


'''
def plot_loss(W, b, train_x, train_y, val_x,val_y, test_x, test_y, alpha, epochs, reg, error_tol):
    
    train_w, train_b = grad_descent(W, b, train_x, train_y, alpha, epochs, reg, error_tol, ModelType.TrainModel)
    val_w, val_b = grad_descent(train_w, train_b, val_x, val_y, alpha, epochs, reg, error_tol, ModelType.ValidationModel)
    grad_descent(val_w, val_b, test_x, test_y, alpha, epochs, reg, error_tol, ModelType.TestModel)
'''


def crossEntropyLoss(W, b, x, y, reg):
    Loss=0
    N= y.shape[0]
    W_tranpose = np.transpose(W)
    yn=1/(1+np.exp(-np.dot(x,W_tranpose)-b))   
    Loss=np.mean(-y*np.log(yn)-(1-y)*np.log(1-yn))
        
    regularizer=reg/2 * np.square(np.matmul(W, W_tranpose))

    return Loss+np.sum(regularizer)

def gradCE(W, b, x, y, reg):
    W_tranpose = np.transpose(W)
    yn=1/(1+np.exp(-np.matmul(x,W_tranpose)-b))
    N= y.shape[0]
    #gradient with respect to weight
    grad=np.matmul(np.transpose(x),yn-y)  
    grad=grad/N+reg*W

    # gradient with respect to bias
    grad_bias=np.sum(yn-y)/N
  
    return grad, grad_bias


def grad_descent_CE(W, b, train_x, train_y, alpha, epochs, reg, error_tol):
    i = 0
    W1 = W2 = W3 = W
    b1 = b2 = b3 = b
    train_loss_set1 = []
    accuracy_set = []

    # while i < epochs and abs(np.linalg.norm(w_step_size))<=error_tol:

    start1 = time.time()
    while i < epochs:
        grad_w1, grad_b1 = gradCE(W1, b1, train_x, train_y, reg)

        w_step_size1 = alpha * grad_w1
        b_step_size1 = alpha * grad_b1
        W1 = W1 - w_step_size1
        b1 = b1 - b_step_size1
        # break

        train_ce1 = crossEntropyLoss(W1, b1, train_x, train_y, reg)
        train_loss_set1.append(train_ce1)

        if abs(np.linalg.norm(w_step_size1)) < error_tol:
            break
        i = i + 1
    start2 = time.time()

    print ("Time: ", start2 - start1)
    W_tranpose = np.transpose(W)
    y_predicted = 1/(1+np.exp(-np.matmul(x, W_tranpose)-b))
    accuracy_set.append(accuracy(y_predicted, train_y))
    print("Accuracy: ",accuracy(y_predicted, train_y))

    #print(start2-start1, start3-start2, start4-start3)
    print("Trainning model\n")
    plt.figure(1)
    plt.title("CE loss set: ephocs=5000 reg=0.1")
    plt.plot(train_loss_set1)
    pic_name = "CE_training_loss.png"
    plt.savefig(pic_name)
    plt.xlabel("epochs")
    plt.ylabel("loss")

    plt.figure(2)
    plt.title("CE accur set: ephocs=5000 reg=0.1")
    plt.plot(accuracy_set)
    pic_name = "CE_training_accuracy.png"
    plt.savefig(pic_name)
    plt.xlabel("accuracy")
    plt.ylabel("loss")

    return W, b



def buildGraph(beta=None, epsilon=None, lossType=None, learning_rate=None):
    # def buildGraph(lossType=None):

    W = tf.Variable(tf.truncated_normal(
        shape=[784, 1], stddev=0.5), name='weights')
    b = tf.Variable(tf.zeros(1), name='biases')
    X = tf.placeholder(tf.float32, [None, 784], name='input_x')
    y_target = tf.placeholder(tf.float32, [None, 1], name='target_y')
    beta = 0
    # Graph definition
    y_predicted = tf.matmul(X, W) + b
    reg_term = beta * tf.nn.l2_loss(W)/2
    tf.set_random_seed(421)
    loss = 0.0

    if lossType == "MSE":
        loss_term = tf.reduce_mean(tf.reduce_mean(tf.square(y_predicted - y_target),
                                                  reduction_indices=1,
                                                  name='squared_error'),
                                   name='mean_squared_error')
        loss = loss_term/2 + reg_term

    elif lossType == "CE":
        logits = tf.matmul(X, W) + b
        y_hat = tf.sigmoid(logits)
        loss_term = tf.losses.sigmoid_cross_entropy(y_target, y_hat)
        loss = loss + reg_term

    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    train = optimizer.minimize(loss)

    return W, b, X, y_target, y_predicted, loss, train


def accuracy(predictions, labels):
    return (np.sum((predictions >= 0.5) == labels) / 500)

# def grad_descent(W, b, trainingData, trainingLabels, alpha, iterations, reg, EPS):
    # Your implementation here

# def buildGraph(beta1=None, beta2=None, epsilon=None, lossType=None, learning_rate=None):
    # Your implementation here


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
    # print (trainData.shape) (3500, 28, 28) 28*28 = 784
    #print (validData.shape) (100, 28, 28)
    #print (testData.shape) (145, 28, 28)

    #print (trainTarget.shape)
    #print (trainTarget.shape[1])
    #print (testData.shape)
    #print (validData.shape)
    train_X = trainData.reshape(3500, 784)
    val_X = validData.reshape(100, 784)
    test_X = testData.reshape(145, 784)

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
    #grad_descent(W, b, train_X, trainTarget, val_X, validTarget, test_X,
    #             testTarget, alpha=0.005, epochs=5000, reg=0, error_tol=0.0000001)

    grad_descent_CE(W, b, train_X, trainTarget, alpha=0.01, epochs=5000, reg=0.1, error_tol=0.0000001)
    #new_W, new_b = grad_descent(W,b,train_X,y=trainTarget,alpha=0.005,epochs=5000,reg=0,error_tol=0.0000001)
    # mse()
    # return
