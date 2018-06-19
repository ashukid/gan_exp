import tensorflow as tf
import numpy as np

def conv2d(x,filter_size,in_channel,out_channel,stride,name):
    # default filter size (5,5)
    with tf.variable_scope(name):
        w = tf.get_variable(name='kernel',shape=(filter_size,filter_size,in_channel,out_channel),initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable(name='biases',shape=(out_channel,),initializer=tf.constant_initializer())
        layer = tf.nn.conv2d(x,filter=w,strides=[1,stride,stride,1],padding='SAME')+b
    
    return layer

def linear(x,out_neuron,name):
    with tf.variable_scope(name):
        w = tf.get_variable(name='weights',shape=(x.get_shape()[1].value,out_neuron),initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable(name='biases',shape=(out_neuron,),initializer=tf.constant_initializer())
        layer = tf.matmul(x,w)+b
    return layer

def flatten(x):
    layer = tf.reshape(x,shape=(-1,x.get_shape()[1:4].num_elements()))
    return layer

def relu(x):
    return tf.nn.relu(x)

def lrelu(x):
    return tf.nn.leaky_relu(x)

def sigmoid(x):
    return tf.nn.sigmoid(x)

def softmax(x):
    return tf.nn.softmax(x)

def tanh(x):
    return tf.nn.tanh(x)

def batch_norm(x):
    layer = tf.layers.batch_normalization(x)
    return layer

def deconv2d(x,in_channel,out_channel,stride):
    w = tf.Variable(tf.random_normal(shape=(5,5,out_channel,in_channel),stddev=0.02))
    out_w = x.get_shape()[1].value*stride
    out_h = x.get_shape()[2].value*stride
    out_shape = [x.get_shape()[0].value,out_w,out_h,out_channel]
    out_shape = tf.stack(out_shape)
    layer = tf.nn.conv2d_transpose(x,filter=w,output_shape=out_shape,strides=[1,stride,stride,1],padding='SAME')
    return layer