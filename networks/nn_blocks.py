from keras.layers import *
from keras.layers.advanced_activations import LeakyReLU
from .instance_normalization import InstanceNormalization
from .GroupNormalization import GroupNormalization
from .pixel_shuffler import PixelShuffler
from .custom_layers.scale_layer import Scale
from .custom_inits.icnr_initializer import icnr_keras
import tensorflow as tf

conv_init = 'he_normal'
w_l2 = 1e-4

def self_attn_block(inp, nc, squeeze_factor=8):
    '''
    Code borrows from https://github.com/taki0112/Self-Attention-GAN-Tensorflow
    '''
    assert nc//squeeze_factor > 0, f"Input channels must be >= {squeeze_factor}, recieved nc={nc}"
    x = inp
    shape_x = x.get_shape().as_list()
    
    f = Conv2D(nc//squeeze_factor, 1)(x)
    g = Conv2D(nc//squeeze_factor, 1)(x)
    h = Conv2D(nc, 1, kernel_initializer=conv_init)(x)
    
    shape_f = f.get_shape().as_list()
    shape_g = g.get_shape().as_list()
    shape_h = h.get_shape().as_list()
    flat_f = Reshape((-1, shape_f[-1]))(f)
    flat_g = Reshape((-1, shape_g[-1]))(g)
    flat_h = Reshape((-1, shape_h[-1]))(h)   
    
    s = Lambda(lambda x: tf.matmul(x[0], x[1], transpose_b=True))([flat_g, flat_f])

    beta = Softmax(axis=-1)(s)
    o = Lambda(lambda x: tf.matmul(x[0], x[1]))([beta, flat_h])
    o = Reshape(shape_x[1:])(o)
    o = Scale()(o)
    
    out = add([o, inp])
    return out

def normalization(inp, norm='none', group='16'):    
    x = inp
    if norm == 'layernorm':
        x = GroupNormalization(group=group)(x)
    elif norm == 'batchnorm':
        x = BatchNormalization()(x)
    elif norm == 'groupnorm':
        x = GroupNormalization(group=16)(x)
    elif norm == 'instancenorm':
        x = InstanceNormalization()(x)
    else:
        x = x
    return x

def conv_block(input_tensor, f, use_norm=False, strides=2, w_l2=w_l2, norm='none'):
    x = input_tensor
    x = Conv2D(f, kernel_size=3, strides=strides, kernel_regularizer=regularizers.l2(w_l2),  
               kernel_initializer=conv_init, use_bias=False, padding="same")(x)
    x = Activation("relu")(x)
    x = normalization(x, norm, f) if use_norm else x
    return x

def conv_block_d(input_tensor, f, use_norm=False, w_l2=w_l2, norm='none'):
    x = input_tensor
    x = Conv2D(f, kernel_size=4, strides=2, kernel_regularizer=regularizers.l2(w_l2), 
               kernel_initializer=conv_init, use_bias=False, padding="same")(x)
    x = LeakyReLU(alpha=0.2)(x)   
    x = normalization(x, norm, f) if use_norm else x
    return x

def res_block(input_tensor, f, use_norm=False, w_l2=w_l2, norm='none'):
    x = input_tensor
    x = Conv2D(f, kernel_size=3, kernel_regularizer=regularizers.l2(w_l2), 
               kernel_initializer=conv_init, use_bias=False, padding="same")(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = normalization(x, norm, f) if use_norm else x
    x = Conv2D(f, kernel_size=3, kernel_regularizer=regularizers.l2(w_l2), 
               kernel_initializer=conv_init, use_bias=False, padding="same")(x)
    x = add([x, input_tensor])
    x = LeakyReLU(alpha=0.2)(x)
    x = normalization(x, norm, f) if use_norm else x
    return x

def upscale_ps(input_tensor, f, use_norm=False, w_l2=w_l2, norm='none'):
    x = input_tensor
    x = Conv2D(f*4, kernel_size=3, kernel_regularizer=regularizers.l2(w_l2), 
               kernel_initializer=icnr_keras, padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = normalization(x, norm, f) if use_norm else x
    x = PixelShuffler()(x)
    return x
