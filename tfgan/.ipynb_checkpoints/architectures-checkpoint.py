class DCGAN:
    
    def __init__(self):
        pass
        
    
    def Generator(self,z,output_h,output_w,output_channel,reuse=False,name='generator'):
        
        assert output_h == output_w
        gf = int(output_h//16)
        stride = 2
        filter_shape = 5
        
        with tf.variable_scope(name):
        
            # no batch norm to generator output layer
            layer1 = linear(z,gf*gf*5*8)
            layer1 = tf.reshape(layer1,shape=(-1,gf,gf,filter_shape*8))
            layer1 = relu(batch_norm(layer1))
            layer2 = relu(batch_norm(deconv2d(layer1,filter_shape*8,filter_shape*4,stride)))
            layer3 = relu(batch_norm(deconv2d(layer2,filter_shape*4,filter_shape*2,stride)))
            layer4 = relu(batch_norm(deconv2d(layer3,filter_shape*2,filter_shape,stride)))
            layer5 = tanh(deconv2d(layer4,filter_shape,output_channel,stride))

            return layer5
        
        
    
    def Descriminator(self,image,reuse=False,name='discriminator'):
        
        
        stride = 2
        filter_channel = 32
        
        with tf.variable_scope(name):
        
            # no batch norm to discriminator input layer
            layer1 = lrelu(conv2d(image,image.get_shape()[-1].value,filter_channel,stride))
            layer2 = lrelu(batch_norm(conv2d(layer1,filter_channel,filter_channel*2,stride)))
            layer3 = flatten(lrelu(batch_norm(conv2d(layer2,filter_channel*2,filter_channel*4,stride))))
            layer4 = linear(layer3,1024)
            output = linear(layer4,1)

            return output,sigmoid(output)
        

