import tensorflow as tf
import numpy as np
from ops import *


#####################################################################################################################
# DCGAN architecture
class DCGAN:
    
    def __init__(self,image_size,image_channel,output_size,output_channel,z_shape):
        self.image_size = image_size
        self.image_channel = image_channel
        self.output_size = output_height
        self.output_channel = output_channel
        self.z_shape = z_shape
    
    
    def Discriminator(self,image,reuse=False,name='discriminator'):
        
        
        maxpool_stride = 2
        normal_stride = 1
        df_size = 5 # dsicriminator filter
        df_channel = 32

        image = tf.reshape(image, [-1,self.image_size,self.image_size,self.image_channel])
        
        with tf.variable_scope(name,reuse=reuse):
        
            l1 = lrelu(conv2d(X,df_size,self.image_channel,df_channel,maxpool_stride,'l1'))
            l2 = lrelu(conv2d(l1,df_size,df_channel,df_channel*2,maxpool_stride,'l2'))
            l3 = lrelu(conv2d(l2,df_size,df_channel*2,df_channel*4,maxpool_stride,'l3'))
            l4 = lrelu(conv2d(l3,l3.get_shape()[1],df_channel*4,1024,normal_stride,'l4','valid')) # valid padding
            l5 = conv2d(l4,1,1024,1,normal_stride,'l5','valid') # valid padding
            output = tf.reshape(l5,shape=[-1,1*1*1])
        
        return output
    
    def Discriminator_bn(self,image,reuse=False,name='discriminator'):
        
        
        maxpool_stride = 2
        normal_stride = 1
        df_size = 5 # dsicriminator filter
        df_channel = 32

        image = tf.reshape(image, [-1,self.image_size,self.image_size,self.image_channel])
        
        with tf.variable_scope(name,reuse=reuse):
            l1 = lrelu(conv2d(X,df_size,self.image_channel,df_channel,maxpool_stride,'l1'))
            l2 = lrelu(batch_norm(conv2d(l1,df_size,df_channel,df_channel*2,maxpool_stride,'l2')))
            l3 = lrelu(batch_norm(conv2d(l2,df_size,df_channel*2,df_channel*4,maxpool_stride,'l3')))
            l4 = lrelu(batch_norm(conv2d(l3,l3.get_shape()[1],df_channel*4,1024,normal_stride,'l4','valid'))) # valid padding
            l5 = conv2d(l4,1,1024,1,normal_stride,'l5','valid') # valid padding
            output = tf.reshape(l5,shape=[-1,1*1*1])

        return output
           
        
    
    def Generator(self,z,reuse=False,name='generator'):
        
        
        gf_size = image_size //16 # generator filter
        gf_channel = 40
        
        with tf.variable_scope(name,reuse=reuse):
            l1 = relu(linear(z,gf_size*gf_size*gf_channel,'l1'))
            l1 = tf.reshape(l1,[-1,gf_size,gf_size,gf_channel])
            l2 = relu(deconv2d(l1,gf_channel,gf_channel//2,2,'l2'))
            l3 = relu(deconv2d(l2,gf_channel//2,gf_channel//4,2,'l3'))        
            l4 = relu(deconv2d(l3,gf_channel//4,gf_channel//8,2,'l4'))  
            l5 = tanh(deconv2d(l4,gf_channel//8,image_channel,2,'l5'))

            return l5
        
    def Generator_bn(self,z,reuse=False,name='generator'):
        
        
        gf_size = image_size //16 # generator filter
        gf_channel = 40
        
        with tf.variable_scope(name,reuse=reuse):
            l1 = relu(batch_norm(linear(z,gf_size*gf_size*gf_channel,'l1')))
            l1 = tf.reshape(l1,[-1,gf_size,gf_size,gf_channel])
            l2 = relu(batch_norm(deconv2d(l1,gf_channel,gf_channel//2,2,'l2')))
            l3 = relu(batch_norm(deconv2d(l2,gf_channel//2,gf_channel//4,2,'l3')))        
            l4 = relu(batch_norm(deconv2d(l3,gf_channel//4,gf_channel//8,2,'l4')))  
            l5 = tanh(deconv2d(l4,gf_channel//8,image_channel,2,'l5'))

            return l5
    
        
#####################################################################################################################
# todo 
# other architectures
# like resnet