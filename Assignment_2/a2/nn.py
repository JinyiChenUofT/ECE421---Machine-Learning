'''
import tensorflow as tf
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

from google.colab import files
files.upload()
'''
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from starter import loadData, convertOneHot, shuffle

trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
train_y, valid_y, test_y = convertOneHot(trainTarget, validTarget, testTarget)

s, w, h = trainData.shape #10000 samples, 28, 28
m,n = train_y.shape
#print(s,w,h)
#print(m,n)

class_names = ['A','B','C','D','E','F','G','H','I','J']

#The number of channels is set to 1 if the image is in grayscale and if the image is in RGB format, the number of channels is set to 3.
s_train,w,h = trainData.shape
train_x = trainData.reshape(-1,w,h,1)
test_x = testData.reshape(-1,w,h,1)
valid_x = validData.reshape(-1,w,h,1)


weights = {
    'w_con': tf.get_variable("w_con", shape=[3,3,1,32],
                initializer=tf.contrib.layers.xavier_initializer()),
    'w_fc1': tf.get_variable("w_fc1", shape=[14*14*32,784],
                initializer=tf.contrib.layers.xavier_initializer()),
    'w_fc2': tf.get_variable("w_fc2", shape=[784,10],
                initializer=tf.contrib.layers.xavier_initializer())
}
biases = {
    'b_con': tf.get_variable('b_con', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
    'b_fc1': tf.get_variable('b_fc1', shape=(784), initializer=tf.contrib.layers.xavier_initializer()),
    'b_fc2': tf.get_variable('b_fc2', shape=(10), initializer=tf.contrib.layers.xavier_initializer())
}
#
x = tf.placeholder(tf.float32, [None, w,h,1])
y = tf.placeholder(tf.float32, [None, n])

def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x) 

def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],padding='SAME')

def batch(x):
    batch_mean, batch_var = tf.nn.moments(x, axes=[0, 1, 2])
    batch_layer = tf.nn.batch_normalization(x,batch_mean,batch_var,offset=None,scale=None,variance_epsilon=1e-3)
    return batch_layer

def conv_net(x, weights, biases, beta=0):  

    #1st 3x3 convolutional layer + relu 
    conv_layer = conv2d(x, weights['w_con'], biases['b_con'])
    
    if (beta!=0):
        regularizer = tf.nn.l2_loss(weights['w_con'])
        conv_layer = tf.reduce_mean(conv_layer + beta * regularizer)
    
    #batch normalization layer
    batch_layer = batch(conv_layer)

    #2x2 max pooling layer
    pool_layer = maxpool2d(batch_layer, k=2)

    #flatten layer
    flatten_layer = tf.reshape(pool_layer, [-1, weights['w_fc1'].get_shape().as_list()[0]])
    #flatten_layer = tf.contrib.layers.flatten(pool_layer)
    
    if (beta!=0):
        regularizer = tf.nn.l2_loss(weights['w_fc1'])
        flatten_layer = tf.reduce_mean(flatten_layer + beta * regularizer)
      
    #1st fc_layer + relu
    fc_layer1 = tf.add(tf.matmul(flatten_layer, weights['w_fc1']), biases['b_fc1'])
    fc_layer1 = tf.nn.relu(fc_layer1)
    
    #2nd fc_layer
    fc_layer2 = tf.add(tf.matmul(fc_layer1, weights['w_fc2']), biases['b_fc2'])

    #softmax output
    out = tf.nn.softmax(fc_layer2)
    print ("out.shape: ",out.shape)
    #cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=out, labels=y))

    return out


def train_model(learning_rate=1e-4,training_iters=50,batch_size=32, beta=0):
    pred = conv_net(x, weights, biases)
    #print ("pred.shape: ", pred.shape)
    #print ("y.shape: ",y.shape)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    if (beta!=0):
        regularizer = tf.nn.l2_loss(weights['w_fc2'])
        cost = tf.reduce_mean(cost + beta * regularizer)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))

    #calculate accuracy across all the given images and average them out. 
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init) 
        train_loss = []
        test_loss = []
        train_accuracy = []
        test_accuracy = []
        valid_loss = []
        valid_accuracy = []
        summary_writer = tf.summary.FileWriter('./Output', sess.graph)
        for i in range(training_iters):
            train_X,train_Y = shuffle(train_x,train_y)
            for batch in range(int(len(train_X)/batch_size)):
                #offset = (batch * batch_size) % (len(train_x) - batch_size)
                #print ("offset: ",offset)
                #print ("offset+batch: ",offset + batch_size)
                #batch_x = train_x[offset:(offset + batch_size), :,:,:]
                #batch_y = train_y[offset:(offset + batch_size), :]
                batch_x = train_X[batch*batch_size:min((batch+1)*batch_size,len(train_X))]
                batch_y = train_Y[batch*batch_size:min((batch+1)*batch_size,len(train_Y))]
                #print("batch_x.shape: ", batch_x.shape)
                #print("batch_y.shape: ",batch_y.shape) 
                # Run optimization op (backprop).
                    # Calculate batch loss and accuracy
                opt = sess.run(optimizer, feed_dict={x: batch_x,
                                                                y: batch_y})
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,y: batch_y})
            test_acc,test_l = sess.run([accuracy,cost], feed_dict={x: test_x, y : test_y})
            valid_acc,valid_l = sess.run([accuracy,cost], feed_dict={x: valid_x, y : valid_y})
            train_loss.append(loss)
            test_loss.append(test_l)
            valid_loss.append(valid_l)
            train_accuracy.append(acc)
            test_accuracy.append(test_acc)
            valid_accuracy.append(valid_acc)

            #print("Testing Accuracy:","{:.5f}".format(test_acc))

            if (i%10==0):
                print("Iter " + str(i) + ", Loss= " + \
                            "{:.6f}".format(loss) + ", Training Accuracy= " + \
                            "{:.5f}".format(acc))
        print("Optimization Finished.")
        summary_writer.close()
        '''
        f = open('data.txt','w')
        f.write("train loss")
        f.write(train_loss)
        f.write("test loss")
        f.write(test_loss)
        f.write("train accuracy")
        f.write(train_accuracy)
        f.write("test accuracy")
        f.write(test_accuracy)
        '''
        plt.figure(1)
        plt.plot(train_loss, 'b', label='Training loss')
        plt.plot(test_loss, 'r', label='Test loss')
        plt.plot(valid_loss, 'g', label='Test loss')
        plt.title('Training and Test loss')
        plt.xlabel('Epochs ',fontsize=16)
        plt.ylabel('Loss',fontsize=16)
        plt.legend()
        plt.show()

        plt.figure(2)
        plt.plot(train_accuracy, 'b', label='Training Accuracy')
        plt.plot(test_accuracy, 'r', label='Test Accuracy')
        plt.plot(valid_accuracy, 'g', label='Test Accuracy')
        #plt.plot(range(len(train_loss)), train_accuracy, 'b', label='Training Accuracy')
        #plt.plot(range(len(train_loss)), test_accuracy, 'r', label='Test Accuracy')
        plt.title('Training and Test Accuracy')
        plt.xlabel('Epochs ',fontsize=16)
        plt.ylabel('Accuracy',fontsize=16)
        plt.legend()
        plt.show()

if __name__ == "__main__":
    #print(len(train_x))
    #print (train_x.shape)
    #print (train_y.shape)
    train_model()
    '''
    batch_x = train_x[0:32]
    batch_y = train_y[0:32]
    print("batch_x.shape: ", batch_x.shape)
    print("batch_y.shape: ",batch_y.shape)
    x = tf.cast(batch_x,tf.float32)
    y = tf.cast(batch_y,tf.float32)
    pred = conv_net(x, weights, biases)
    print ("pred.shape: ", pred.shape)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
    '''