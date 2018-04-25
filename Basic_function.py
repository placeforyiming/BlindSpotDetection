import tensorflow as tf
from tflearn.layers.conv import global_avg_pool
from tensorflow.contrib.layers import batch_norm, flatten
from tensorflow.contrib.framework import arg_scope

regularizer=tf.contrib.layers.l2_regularizer(scale=0.0001)

#followed with a relu
def conv_layer(input, filter, kernel, stride=1, padding='SAME', layer_name="conv"):
    with tf.name_scope(layer_name):
        network = tf.layers.conv2d(inputs=input, use_bias=True, filters=filter, kernel_size=kernel, strides=stride, padding=padding,kernel_regularizer=regularizer)
        return network

def conv_layer_Sep(input, filter, kernel, stride=1, padding='SAME', layer_name="conv"):
    with tf.name_scope(layer_name):
        network = tf.layers.separable_conv2d(inputs=input,use_bias=True,filters=filter,kernel_size=kernel,  strides=stride, padding=padding,depth_multiplier=5,depthwise_regularizer=regularizer,pointwise_regularizer=regularizer,trainable=True)
        return network

def Fully_connected(x, out_num=10, layer_name='fully_connected') :
    with tf.name_scope(layer_name) :
        return tf.layers.dense(inputs=x, use_bias=True, units=out_num,kernel_regularizer=regularizer)

def Relu(x):
    return tf.nn.relu(x)

def Elu(x):
    return tf.nn.elu(x)

def Sigmoid(x):
    return tf.nn.sigmoid(x)

def Global_Average_Pooling(x):
    return global_avg_pool(x, name='Global_avg_pooling')

def Max_pooling(x, pool_size=[3,3], stride=2, padding='VALID') :
    return tf.layers.max_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)

def Avg_pooling(x, pool_size=[3,3], stride=1, padding='SAME') :
    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)

#Get the input x, squeeze with input_channel/R, output is the result of product of probability and input
def SqueezeExcitation(x,input_channel,R,batch_size,trainable,se_name="SE_1"):
    with tf.name_scope(se_name):
        Glo_Pool=Global_Average_Pooling(x)
        assert input_channel==Glo_Pool.get_shape()[1]
        x=Batch_Normalization(x,training=trainable,scope=se_name+"BN")
        FC_1=tf.layers.dense(inputs=Glo_Pool, use_bias=True, units=input_channel/R)
        ReLu_1=Relu(FC_1)
        FC_2=tf.layers.dense(inputs=ReLu_1, use_bias=True, units=input_channel)
        Probability=Sigmoid(FC_2)
        Probability =tf.clip_by_value(Probability, 0.2, 0.99)
        Probability=tf.reshape(Probability,(-1,1,1,input_channel))
        Output=x*Probability
        assert Output.get_shape()[1:4]==x.get_shape()[1:4]
        return Output




def Batch_Normalization(x, training, scope):

    with arg_scope([batch_norm],
                   scope=scope,
                   updates_collections=None,
                   decay=0.9,
                   center=True,
                   scale=True,
                   zero_debias_moving_mean=True) :
        training=tf.constant(training,dtype=tf.bool)
        return tf.cond(training,
                       lambda : batch_norm(inputs=x, is_training=training, reuse=None),
                       lambda : batch_norm(inputs=x, is_training=training, reuse=True))

def Concatenation(layers) :
    return tf.concat(layers, axis=3)

def Dropout(x, rate) :
    return tf.nn.dropout(x=x, keep_prob=rate)

