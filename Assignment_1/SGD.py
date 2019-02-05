import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from starter import buildGraph, loadData

def trainModel(lossType=None):

    # Initialize session
    #X = X.reshape(3500,784)
                
    trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
    trainData = trainData.reshape(3500,784)
    W, b, X, y_target, y_predicted, loss, train = buildGraph(lossType=lossType)
    init = tf.global_variables_initializer()
    sess = tf.InteractiveSession()

    sess.run(init)

    # Training the model
    
    batch_size = 500
    for i in range (700):

        randIndx = np.arange(len(trainData))
        np.random.shuffle(randIndx)
        trainData, trainTarget = trainData[randIndx], trainTarget[randIndx]

        for step in range(7):
            offset = (step * batch_size) % (len(trainData) - batch_size)
            # Generate a minibatch.
            batch_data = trainData[offset:(offset + batch_size), :]
            batch_labels = trainTarget[offset:(offset + batch_size), :]
                        
            _, err, currentW, currentb, predictions = sess.run([train, loss, W, b, y_predicted], feed_dict={X: batch_data, y_target: batch_labels}) 
            #if step % 10 == 0:
            #        print("Iteration: %d, MSE-training: %.2f" %(step, err))
        
    return currentW, currentb, predictions   
    # Final testing error
    #errTest = sess.run(meanSquaredError, feed_dict = {X: testData, y_target: testTarget})
    #print("Final Testing MSE: %.2f:" % (errTest))
def classify_result(Data, Target, W,b):
    N=Target.shape[0]
    classifier=np.zeros((N,1))
    predict=np.dot(Data,W)+b
    loss=0
  
    for i in range(N):
        if predict[i]>0.5:
           classifier[i]=1
        else:
           classifier[i]=0
      
        if Target[i]!=classifier[i]:
           loss=loss+1
      
    acc = 1 - loss/N
    print(classifier.shape)
    print(Target.shape)
    return classifier, loss, acc

def accuracy(predictions, labels):
    
   return (100.00* np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))/predictions.shape[0])
         

'''
def accuracy(predictions, labels):
        return (np.sum((predictions>=0.5)==labels) / np.shape(predictions)[0])
'''
if __name__ == '__main__':
        W, b, predictions = trainModel("CE")
        trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
        validData = validData.reshape(100,784)
        trainData = trainData.reshape(3500,784)
        #accura = accuracy(predictions,trainTarget)
        #print (accura)
        #classifier,loss = classify_result(validData, validTarget, W,b)
        classifier,loss,acc = classify_result(trainData, trainTarget, W,b)
        print ("done")
        print (loss)
        print (acc)
        '''
        classifier,loss = classify_result(trainData, trainTarget, W,b)
        print ("done")
        print(classifier)
        print(loss)
        '''
'''
plt.plot(range(0, 100, 5), loss_batch, 'r--', label='Batch Loss')
plt.plot(range(0, 100, 5), loss_stochastic, 'b-', label='Stochastic Loss')
plt.legend(loc='upper right')
plt.title('Batch training vs Stochastic training')
plt.show()
'''