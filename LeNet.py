#!/usr/bin/env python

import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from tensorflow.contrib.layers import flatten
import os
import gc
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def load_dataset():
    from tensorflow.examples.tutorials.mnist import input_data
    #https://www.tensorflow.org/versions/r1.1/get_started/mnist/beginners
    mnist = input_data.read_data_sets("MNIST_data/", reshape=False)
    X_train, y_train = mnist.train.images, mnist.train.labels
    X_validation, y_validation = mnist.validation.images, mnist.validation.labels
    X_test, y_test = mnist.test.images, mnist.test.labels

    assert(len(X_train) == len(y_train));
    assert(len(X_validation) == len(y_validation));
    assert(len(X_test) == len(y_test));

    #print ("Image shape: ", X_train[0].shape)
    #print (" Training size: ", len(X_train))
    #print (" Validation size: ", len(X_validation))
    #print (" Test size: ", len(X_test))
    return X_train, X_validation, X_test, y_train, y_validation, y_test

def convert_leNet(X_train, X_validation, X_test):
    #https://medium.com/@siddharthdas_32104/cnns-architectures-lenet-alexnet-vgg-googlenet-resnet-and-more-666091488df5
    #LeNet accepts 32*32*C; where C is channels
    # npad is a tuple of (n_before, n_after) for each dimension
    npad = ((0,0), (2,2), (2,2),(0,0))
    X_train = np.pad(X_train, npad, mode='constant', constant_values=0)
    X_validation = np.pad(X_validation, npad, mode='constant', constant_values=0)
    X_test = np.pad(X_test, npad, mode='constant', constant_values=0)

    #print ("shape now is :", X_train[0].shape)

    return X_train, X_validation, X_test


def convolution(f, s, n_c, layer, input_conv):
    mu = 0;
    sigma = 0.1;
    conv_w = tf.Variable(tf.truncated_normal(shape= [f,f,n_c, layer], mean = mu, stddev=sigma))
    #bias
    conv_b = tf.Variable(tf.zeros(layer))
    conv = tf.nn.conv2d(input_conv, conv_w, strides = [s,s,s,s], padding='VALID') + conv_b
    return conv

def ReLU(conv):
    return tf.nn.relu(conv)

def max_pool(f, s, input_pool):
    #https://stackoverflow.com/questions/38601452/the-usages-of-ksize-in-tf-nn-max-pool
    pool = tf.nn.max_pool(input_pool, ksize=[1,f, f,1], strides=[1,s,s,1], padding='VALID')
    return pool

def fully_connected(input_fully, size_input_fully, layer):
    mu = 0;
    sigma = 0.1;
    FC_w = tf.Variable(tf.truncated_normal(shape= [size_input_fully, layer], mean = mu, stddev=sigma))
    #bias
    FC_b = tf.Variable(tf.zeros(layer))
    fc = tf.matmul(input_fully, FC_w) + FC_b
    #activation
    return fc

def output_matrix(n, p, f, s):
    return 1 + (n + 2*p - f)/s

def LeNet(x):
    #https://cdn-images-1.medium.com/max/800/0*MU7G1aH1jw-6eFiD.png
    #https://paper.dropbox.com/doc/deeplearning.ai-WcRJPTHA2vYx86v6hbMOK - and refer to notes here.
    #Hyper-parameters:
    layer_depth = {
            'layer_1': 6,
            'layer_2': 16,
            'layer_3': 120,
            'layer_FC': 84
            }

    strides = {
            's_1': 1,
            's_2': 2,
            's_3': 1,
            's_4': 2,
            }
    filters = {
            'f_1': 5,
            'f_2': 2,
            'f_3': 5,
            'f_4': 2
            }
    padding = {
            'p_1': 0,
            'p_2': 0,
            'p_3': 0,
            'p_4': 0
            }

    #XXX: Layer-1: Convolutional Input=32*32*1 and Output = 28*28*6
    # number of channels
    n_c = int(x.shape[-1])
    conv1 = convolution(filters['f_1'],strides['s_1'], n_c, layer_depth['layer_1'], x)
    conv1 = ReLU(conv1)
    #XXX: Max Pooling -> Select max value in the pool
    #Input = 28*28*6; Output = 14*14*6
    pool_1 = max_pool(filters['f_2'], strides['s_2'], conv1)

    #XXX: Conv2, Input=14*14*6; Output=10*10*16
    n_c = int(pool_1.shape[-1])
    conv2 = convolution(filters['f_3'], strides['s_3'], n_c, layer_depth['layer_2'], pool_1)
    conv2 = ReLU(conv2)

    #XXX: Max pooling
    #Input = 10*10*16; output = 5*5*16
    pool_2 = max_pool(filters['f_4'], strides['s_4'], conv2)

    #XXX: Flatten
    #Input = 5 * 5 * 16; output = 400 * 1
    FC1 = flatten(pool_2)

    #XXX: Fully connected, Input = 400*1; Output = 120 * 1
    #All 400 neurons are connected with 120
    size_input_fully = FC1.get_shape().as_list()[1]
    fc1 = fully_connected(FC1, size_input_fully, layer_depth['layer_3'])
    fc1 = ReLU(fc1)

    #XXX: Fully connected, Input = 120*1; Output = 84 * 1
    #All 120 neurons are connected with 84
    fc2 = fully_connected(fc1, layer_depth['layer_3'], layer_depth['layer_FC'])
    fc2 = ReLU(fc2)

    #XXX: Fully connected Input=84*1; Ouput = 10*1
    y_hat = fully_connected(fc2, layer_depth['layer_FC'], 10)
    return y_hat

def evaluate(X_data, y_data, accuracy_operation, sess):

    m = len(X_data)
    total_accuracy = 0
    for offset in range(0, m, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / m

def main():
    X_train, X_validation, X_test, y_train, y_validation, y_test = load_dataset()
    X_train, X_validation, X_test = convert_leNet(X_train, X_validation, X_test)
    #shuffle
    X_train, y_train = shuffle(X_train, y_train)

    #Global parameters
    global BATCH_SIZE
    EPOCHS = 10
    BATCH_SIZE = 128


    #features and labels
    global x, y
    x = tf.placeholder(tf.float32, (None, 32, 32, 1)) # it's an image
    y = tf.placeholder(tf.int32, (None)) # a single value

    #https://stackoverflow.com/questions/33681517/tensorflow-one-hot-encoder
    #convert a set of sparse labels to a dense one-hot representation
    one_hot_y = tf.one_hot(y, 10)

    #train
    rate = 0.001
    y_hat = LeNet(x)

    cross_entropy  = tf.nn.softmax_cross_entropy_with_logits(logits=y_hat, labels=one_hot_y, name=None)
    loss_operation = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate = rate)
    training_operation = optimizer.minimize(loss_operation)

    correct_prediction = tf.equal(tf.argmax(y_hat, 1), tf.argmax(one_hot_y, 1))
    accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    #Train the model
    sess = tf.Session()#config=tf.ConfigProto(log_device_placement=True))
    #https://www.tensorflow.org/programmers_guide/using_gpu
    #sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    m = len(X_train)

    print ("training...")
    for i in range(EPOCHS):
        for offset in range(0, m, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})

        validation_accuracy = evaluate(X_validation, y_validation, accuracy_operation, sess)
        print("EPOCH: ", i+1)
        print("Validation Accuracy %0.3f" % (validation_accuracy))

    test_accuracy = evaluate(X_test, y_test, accuracy_operation, sess)
    print("Test Accuracy = {:.3f}".format(test_accuracy))
    sess.close()

if __name__ == '__main__':
    main()
