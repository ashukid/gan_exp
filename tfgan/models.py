# wasserstein gan implementation
class WassGan:
    
    def __init__(self,D_real,D_fake,d_var_scope='discriminator',g_var_scope='generator'):
        allvars = tf.trainable_variables()
        self.dvars = [var for var in allvars if d_var_scope in var.name]
        self.gvars = [var for var in allvars if g_var_scope in var.name]
        self.D_real = D_real
        self.D_fake = D_fake
    
    
    def loss(self):
        with tf.name_scope('losses'):
            # minimize both loss
            dloss = -(tf.reduce_mean(self.D_real)-tf.reduce_mean(self.D_fake))
            gloss = -(tf.reduce_mean(self.D_fake))
            clip_d = [p.assign(tf.clip_by_value(p,-0.01,0.01)) for p in self.dvars]
        
            return dloss,gloss,clip_d
        
        
    def optimizers(self,d_lr=0.005,g_lr=0.005):
        
        with tf.variable_scope('optimizers',reuse=tf.AUTO_REUSE):
            doptimizer = tf.train.RMSPropOptimizer(learning_rate=d_lr).minimize(dloss,var_list=dself.vars)
            goptimizer = tf.train.RMSPropOptimizer(learning_rate=g_lr).minimize(gloss,var_list=self.gvars)
            
            return doptimizer,goptimizer
        
# improved wasserstein gan implementation
class WassGanImproved:
    
    def __init__(self,D_real,D_fake,d_var_scope='discriminator',g_var_scope='generator'):
        allvars = tf.trainable_variables()
        self.dvars = [var for var in allvars if d_var_scope in var.name]
        self.gvars = [var for var in allvars if g_var_scope in var.name]
        self.D_real = D_real
        self.D_fake = D_fake
        
    def loss(self):
        with tf.name_scope('losses'):
            
            
    def optimizers(self,epsilon=0.0001,beta1=0,beta2=0.9):
        with tf.variable_scope('optimizers',reuse=tf.AUTO_REUSE):
            