"""
@author     : Qizhi He @ PNNL (qizhi.he@pnnl.gov)
"""
import tensorflow as tf

def tf_session():
    # tf session
    config = tf.ConfigProto(allow_soft_placement=True,
                            log_device_placement=True)
    config.gpu_options.force_gpu_compatible = True
    sess = tf.Session(config=config)
    # self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=True)) #standard setting
    ## init
    init = tf.global_variables_initializer()
    sess.run(init)
    
    return sess

