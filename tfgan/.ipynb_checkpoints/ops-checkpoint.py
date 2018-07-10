import tensorflow as tf
import numpy as np

def conv2d(x,filter_size,in_channel,out_channel,stride,name,padding='same'):
    # default filter size (5,5)
    with tf.variable_scope(name):
        w = tf.get_variable(name='kernel',shape=[filter_size,filter_size,in_channel,out_channel],
                            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable(name='biases',shape=[out_channel],initializer=tf.constant_initializer())
        layer = tf.nn.conv2d(x,filter=w,strides=[1,stride,stride,1],padding=padding.upper())+b
    
    return layer

def linear(x,out_neuron,name):
    with tf.variable_scope(name):
        w = tf.get_variable(name='weights',shape=[x.get_shape()[1].value,out_neuron],
                            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable(name='biases',shape=[out_neuron],initializer=tf.constant_initializer())
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
    layer = tf.contrib.layers.batch_norm(x)
    return layer

def deconv2d(x,in_channel,out_channel,stride,name,):
    with tf.variable_scope(name):
        w = tf.get_variable(name='kernel',shape=[5,5,out_channel,in_channel],
                            initializer=tf.contrib.layers.xavier_initializer())
        out_w = x.get_shape()[1].value*stride
        out_h = x.get_shape()[2].value*stride
        if(x.get_shape()[0] == None):
            print("Placeholder Error : Define placeholder with batch_size instead of 'None' ")
            return None
        # outshape needs exact integers instead on None or -1
        out_shape = [x.get_shape()[0].value,out_w,out_h,out_channel]
        out_shape = tf.stack(out_shape)
        layer = tf.nn.conv2d_transpose(x,filter=w,output_shape=out_shape,
                                       strides=[1,stride,stride,1],padding='SAME')
        return layer

# image helper functions
def convert_to_tanh(im):
    #print("Converting Image in tanh range ... !")
    max_value = np.round(np.max(im))
    min_value = np.round(np.min(im))
    if(max_value == 1 and min_value == -1):
        #print("Array already in tanh range")
        return im
    return np.round(((im/max_value)*2)-1,decimals=2)

def convert_from_tanh(im):
    #print("Converting image from tanh to 0-255 range")
    max_value = np.round(np.max(im))
    min_value = np.round(np.min(im))
    if(max_value == 255 and min_value == 0):
        # print("Array already in 0-255 range")
        return im
    if(max_value == 1 and min_value ==0 ):
        return im*255
    return np.round(((im+1)/2)*max_value,decimals=2)