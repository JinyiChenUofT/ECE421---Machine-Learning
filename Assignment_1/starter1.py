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

def MSE(W, b, x, y, reg):
   Loss=0
   N= y.shape[0]
   Loss=np.mean(np.square(np.dot(x,W.T)+b-y))
      
   regularizer=reg/2 * np.square(np.matmul(W, W.T))

   return Loss/2+np.sum(regularizer)

def gradMSE(W, b, x, y, reg):
   N=y.shape[0]
   grad=np.zeros((1,784))
   grad=np.dot(x.T,(np.dot(x,W.T)+b-y))
   grad=grad/N+reg*W.T
  
   grad_bias=np.mean((np.dot(x,W.T)+b-y))
   grad=grad.T
   return grad, grad_bias

def crossEntropyLoss(W, b, x, y, reg):
   Loss=0
   N= y.shape[0]
  
   yn=1/(1+np.exp(-np.dot(x,W.T)-b))  
   Loss=np.mean(-y*np.log(yn)-(1-y)*np.log(1-yn))
      
   regularizer=reg/2 * np.square(np.matmul(W, W.T))

   return Loss+np.sum(regularizer)

def gradCE(W, b, x, y, reg):
  
   yn=1/(1+np.exp(-np.matmul(x,W.T)-b))
   N= y.shape[0]
  
   #grad=np.dot(x.T,((2*y-1)*yn-y))
   grad=np.matmul(x.T,yn-y)
   #print(grad.shape)
   grad=grad/N+reg*W.T
   #grad_bias=np.mean((2*y-1)*yn-y)
   grad_bias=np.sum(yn-y)/N
  
   return grad, grad_bias


def grad_descent(W, b, x, y, alpha, epochs, reg, error_tol, lossType):
   N=y.shape[0]
   i=0
   mse_old=1000
   if lossType==0:
       for i in range(epochs):  
           grad_W, grad_bias=gradMSE(W,b,x,y,reg)
           W=W-alpha*grad_W
           b=b-alpha*grad_bias
           mse_new=MSE(W,b,x,y,reg)
  
       return W, b
   else:  
       i=0
       for i in range(epochs):
           grad_log_W, grad_log_bias=gradCE(W,b,x,y,reg)
           W=W-alpha*grad_log_W
           b=b-alpha*grad_log_bias
           #mse=crossEntropyLoss(W,b,x,y,reg)
          
           #if i%100==0:
           print(i)
      
   return W, b

def grad_descent_log(W, b, x, y, alpha, epochs, reg, error_tol, lossType):
   N=y.shape[0]
   i=0
   mse_old=1000  
   for i in range(epochs):  
       grad_log_W, grad_log_bias=gradCE(W,b,x,y,reg)
       W=W-alpha*grad_log_W
       b=b-alpha*grad_log_bias
       print(i)
      
   return W, b

#def buildGraph(beta1=None, beta2=None, epsilon=None, lossType=None, learning_rate=None):
   # Your implementation here
def buildGraph(losstype):
   reg = 0
   batch=100
   graph = tf.Graph()
   learning_rate=0.001
   epochs=700
   trained_W=[]
   trained_b=[]
   shuffled_target=[]
   shuffled_data=[]
   merged_data=[]
   #to plot
   train_loss_set = []
   valid_loss_set = []
   test_loss_set = []
   train_acc_set = []
   valid_acc_set = []
   test_acc_set = []

   with graph.as_default(): 
       W=tf.Variable(tf.truncated_normal(mean=0,shape=[784,1],stddev=0.5))
       b=tf.Variable(0.00)
       x = tf.placeholder(tf.float32,[batch,784])
       y = tf.placeholder(tf.float32,[batch,1])

       validData = tf.placeholder(tf.float32, shape=(100, 784))
       validTarget = tf.placeholder(tf.int8, shape=(100, 1))

       testData = tf.placeholder(tf.float32, shape=(145, 784))
       testTarget = tf.placeholder(tf.int8, shape=(145, 1))

       tf.set_random_seed(421)
       if losstype==0:
           trainPrediction=tf.matmul(x,W)+b
           MSE=tf.reduce_mean(tf.square(trainPrediction-y))+reg*tf.nn.l2_loss(W)
       #loss = tf.losses.mean_squared_error(y, y_p)
           mse=tf.div(MSE,2,name="mse")
          
        
           trainLoss = tf.losses.mean_squared_error(y, trainPrediction)
           regularizer = tf.nn.l2_loss(W)
           trainLoss = trainLoss + reg/2.0 * regularizer
           optimizer = tf.train.AdamOptimizer(learning_rate=0.001,epsilon=1e-04)
           train_op=optimizer.minimize(mse)

           validPrediction = tf.matmul(validData,W)+b
           validLoss = tf.losses.mean_squared_error(validTarget, validPrediction)
           validLoss = validLoss + reg/2.0 * regularizer
           

           testPrediction = tf.matmul(testData,W)+b
           testLoss = tf.losses.mean_squared_error(testTarget, testPrediction)
           testLoss = testLoss + reg/2.0 * regularizer

          
       else:
           #yn=tf.log_sigmoid(tf.matmul(x,W)+b)
           #CE=tf.reduce_mean(-y*tf.log(yn)-(1-y)*tf.log(1-yn))+reg*tf.nn.l2_loss(W)
           #Loss=tf.div(CE,2)
           logits = tf.matmul(x, W) + b
           trainPrediction = tf.sigmoid(logits)
           trainLoss = tf.losses.sigmoid_cross_entropy(y, trainPrediction)
           # Loss function using L2 Regularization
           regularizer = tf.nn.l2_loss(W)
           trainLoss = trainLoss + reg/2.0 * regularizer
      
           optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
           train_op=optimizer.minimize(trainLoss)

           validLogits = tf.matmul(validData, W) + b
           validPrediction = tf.sigmoid(validLogits)
           validLoss = tf.losses.sigmoid_cross_entropy(validTarget, validPrediction)

           testLogits = tf.matmul(testData, W) + b
           testPrediction = tf.sigmoid(testLogits)
           testLoss = tf.losses.sigmoid_cross_entropy(testTarget, testPrediction)

       init=tf.global_variables_initializer()
  
       with tf.Session(graph=graph) as sess:
           sess.run(init)
          
           for i in range(epochs):
               traindata, validdata, testdata, traintarget, validtarget, testtarget = loadData()
               traindata=traindata.reshape(3500,784)
               validdata=validdata.reshape(100,784)
               testdata=testdata.reshape(145,784)
               n=int(3500/batch)
                
                               
               for j in range(n):
                   
                   merged_data=np.append(traindata,traintarget,axis=1)   
                   np.random.shuffle(merged_data)
                   shuffled_target=merged_data[:,784:]
                   shuffled_data=merged_data[:,:784]
                   
                   randIndx = np.arange(len(trainData))
                   np.random.shuffle(randIndx)
                   shuffled_data, shuffled_target = trainData[randIndx], trainTarget[randIndx]
                   
                   X = shuffled_data[j*batch:(j+1)*batch,]
                   Y = shuffled_target[j*batch:(j+1)*batch,]

                   _,trained_W,trained_b,train_prediction,train_loss, valid_loss, test_loss,valid_prediction,test_prediction= sess.run([train_op,W,b,trainPrediction,trainLoss,validLoss,testLoss,validPrediction,testPrediction],feed_dict={x: X,y: Y, validData: validdata, validTarget: validtarget, testData: testdata, testTarget: testtarget})
               
               train_loss_set.append(train_loss)
               valid_loss_set.append(valid_loss)
               test_loss_set.append(test_loss)
               train_acc_set.append(accuracy(train_prediction,Y))
               valid_acc_set.append(accuracy(valid_prediction,validtarget))
               test_acc_set.append(accuracy(test_prediction,testtarget))

               #sess.run(train_op, feed_dict={
                # x: traindata,
                # y: traintarget})

           
            
           print("SGD loss\n")

           print ("train loss:", train_loss_set[-1])
           print ("valid loss:", valid_loss_set[-1])
           print ("test loss:", test_loss_set[-1])    
           plt.title("SGD CE loss set: ephocs=700 B=100")
           plt.plot(train_loss_set, label='train loss')
           plt.plot(valid_loss_set, label='valid loss')
           plt.plot(test_loss_set, label='test loss')
           pic_name = "SGD_MSE_loss_B100.png"
           plt.savefig(pic_name)
           plt.xlabel("epochs")
           plt.ylabel("loss")
           plt.legend(loc='upper right')
           

           print("SGD accuracy\n")
           print ("train acc:", train_acc_set[-1])
           print ("valid acc:", valid_acc_set[-1])
           print ("test acc:", test_acc_set[-1])
           plt.figure(2)
           plt.title("SGD CE accuracy set: ephocs=700 B=100")
           plt.plot(train_acc_set, label='train accuracy')
           plt.plot(valid_acc_set, label='valid accuracy')
           plt.plot(test_acc_set, label='test accuracy')
           pic_name = "SGD_MSE_acc_B100.png"
           plt.savefig(pic_name)
           plt.xlabel("epochs")
           plt.ylabel("accuracy")
           plt.legend(loc='lower right')
           plt.show()

   #print(loss)
   return trained_W, trained_b, y_predict, traintarget, reg
#weight, bias, predicted labels, real labels, the loss, the optimizer and the regularization parameter

def accuracy(predictions, labels):
   return (np.sum((predictions>=0.5)==labels) / np.shape(predictions)[0])


def classify_result(Data, Target, W):
   N=Target.shape[0]
   classifier=np.zeros((N,1))
   predict=np.dot(Data,W.T)
   loss=0
  
   for i in range(N):
       if predict[i]>b:
           classifier[i]=1
       else:
           classifier[i]=0
      
       if Target[i]!=classifier[i]:
           loss=loss+1
      
      
   print(classifier.shape)
   print(Target.shape)
   return classifier, loss

def normal_equation(x, y, reg):
   X=x
   N=y.shape[0]
   d=x.shape[1]
   x_zero=np.ones((N,1))
   I=np.identity(d+1)
   I[0,0]=0
   X=np.append(x_zero,X,axis=1)
   print(X.shape)
   W_star=np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)+reg*I),X.T),y)
   bias=W_star[0]
   W_star=np.delete(W_star,(0),axis=0)
   return W_star, bias



trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()

W=np.ones((1,784))
b=0
trainData=trainData.reshape(3500,784)
validData=validData.reshape(100,784)
testData=testData.reshape(145,784)

#new_W, new_b = grad_descent(W,0,trainData,trainTarget,0.005,5000,0,10**(-7),0)

#W_norm, b_norm=normal_equation(trainData,trainTarget,0)

W_log=np.zeros((1,784))

#logW, logb=grad_descent_log(W_log,0,trainData,trainTarget,0.005,5000,0,1*10**(-7),1)
trained_W, trained_b, y_predict, traintarget, reg= buildGraph(1)
yn=1/(1+np.exp(-np.matmul(trainData,trained_W)-trained_b))
yn_p=np.matmul(trainData,trained_W)+trained_b
print(accuracy(yn, trainTarget))
print(accuracy(yn_p, trainTarget))


