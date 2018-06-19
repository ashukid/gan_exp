import tensorflow as tf
import numpy as np
from ops import *


#####################################################################################################################
# DCGAN architecture
class DCGAN:
    
    def __init__(self,image_height,image_width,image_channel,output_height,output_width,output_channel,z_shape):
        self.image_height = image_height
        self.image_width = image_width
        self.image_channel = image_channel
        self.output_height = output_height
        self.output_width = output_width
        self.output_channel = output_channel
        self.z_shape = z_shape
    
    
    def Discriminator_bn(self,image,reuse=False,name='discriminator'):
        
        
        maxpool_stride = 2
        normal_stride = 1
        filter_channel = 32
        image = tf.reshape(image, [-1,self.image_height,self.image_width,self.image_channel])
        
        with tf.variable_scope(name,reuse=reuse):
        
            # no batch norm to discriminator input layer
            layer1 = lrelu(conv2d(image,5,image.get_shape()[-1].value,filter_channel,maxpool_stride))
            layer2 = lrelu(batch_norm(conv2d(layer1,5,filter_channel,filter_channel*2,maxpool_stride)))
            layer3 = lrelu(batch_norm(conv2d(layer2,5,filter_channel*2,filter_channel*4,maxpool_stride)))
            layer4 = lrelu(batch_norm(conv2d(layer3,self.image_height//8,filter_channel*4,1024,normal_stride)))
            output = lrelu(batch_norm(conv2d(layer4,1,1024,1,normal_stride)))
            output = tf.reshape(output,shape=(-1,1*1*1))

            return output,sigmoid(output)
        
    
    def Generator_bn(self,z,reuse=False,name='generator'):
        
        assert self.output_height == self.output_width
        gf = int(self.output_height//16)
        stride = 2
        filter_shape = 5
        
        with tf.variable_scope(name,reuse=reuse):
        
            # no batch norm to generator output layer
            layer1 = linear(z,gf*gf*5*8)
            layer1 = tf.reshape(layer1,shape=(-1,gf,gf,filter_shape*8))
            layer1 = relu(batch_norm(layer1))
            layer2 = relu(batch_norm(deconv2d(layer1,filter_shape*8,filter_shape*4,stride)))
            layer3 = relu(batch_norm(deconv2d(layer2,filter_shape*4,filter_shape*2,stride)))
            layer4 = relu(batch_norm(deconv2d(layer3,filter_shape*2,filter_shape,stride)))
            layer5 = tanh(deconv2d(layer4,filter_shape,self.output_channel,stride))

            return layer5
        
#####################################################################################################################
# todo 
# other architectures