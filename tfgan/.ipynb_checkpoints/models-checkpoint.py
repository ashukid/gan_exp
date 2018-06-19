import tensorflow as tf
import numpy as np

###############################################################################################################   
# wasserstein gan implementation
class WassGan:
    
    def __init__(self,model,real_image,z):
        
        self.model = model
        self.real_image = real_image
        self.z = z
        
        self.fake_image = self.model.Generator(self.z)
        self.D_real,self.L_real = self.model.Discriminator(self.real_image)
        self.D_fake,self.L_fake = self.model.Discriminator(self.fake_image,reuse=True)
        
        allvars = tf.trainable_variables()
        self.dvars = [var for var in allvars if 'discriminator' in var.name]
        self.gvars = [var for var in allvars if 'generator' in var.name]
    
    def loss(self,c=0.01):
        with tf.name_scope('losses'):
            # minimize both loss
            dloss = -(tf.reduce_mean(self.D_real)-tf.reduce_mean(self.D_fake))
            gloss = -(tf.reduce_mean(self.D_fake))
            clip_d = [p.assign(tf.clip_by_value(p,-c,c)) for p in self.dvars]
        
            return dloss,gloss,clip_d
        
        
    def optimizers(self,dloss,gloss,alpha=0.0005):
        
        with tf.variable_scope('optimizers',reuse=tf.AUTO_REUSE):
            doptimizer = tf.train.RMSPropOptimizer(learning_rate=alpha).minimize(dloss,var_list=self.dvars)
            goptimizer = tf.train.RMSPropOptimizer(learning_rate=alpha).minimize(gloss,var_list=self.gvars)
            
            return doptimizer,goptimizer
        
        
        
################################################################################################################              
# improved wasserstein gan implementation
class WassGanImproved:
    
    def __init__(self,model,real_image,z):
        
        self.model = model
        self.real_image = real_image
        self.z = z
        
        self.fake_image = self.model.Generator(self.z)
        self.D_real,self.L_real = self.model.Discriminator(self.real_image)
        self.D_fake,self.L_fake = self.model.Discriminator(self.fake_image,reuse=True)
        
        allvars = tf.trainable_variables()
        self.dvars = [var for var in allvars if 'discriminator' in var.name]
        self.gvars = [var for var in allvars if 'generator' in var.name]
        
        
    def loss(self,_lambda=10):
        
        with tf.name_scope('losses'):
            dloss = -(tf.reduce_mean(self.D_real)-tf.reduce_mean(self.D_fake))
            gloss = -(tf.reduce_mean(self.D_fake))
            
            alpha= tf.random_uniform(shape=[1], minval=0.,maxval=1.)
            differences = self.fake_image - self.real_image 
            interpolates = self.real_image + (alpha*differences)
            gradients = tf.gradients(self.model.Discriminator(interpolates,name='inter_discriminator'), [interpolates])[0]
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
            gradient_penalty = tf.reduce_mean((slopes-1.)**2)
            dloss += _lambda*gradient_penalty
            
            return dloss,gloss
            
    def optimizers(self,dloss,gloss,alpha=0.0001,beta1=0,beta2=0.9):
        with tf.variable_scope('optimizers',reuse=tf.AUTO_REUSE):
            doptimizer = tf.train.AdamOptimizer(learning_rate=alpha).minimize(dloss,var_list=self.dvars)
            goptimizer = tf.train.AdamOptimizer(learning_rate=alpha).minimize(gloss,var_list=self.gvars)
            
            return doptimizer,goptimizer