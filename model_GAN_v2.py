from keras.models import Sequential, Model
from keras.layers import *
from keras.layers.advanced_activations import LeakyReLU
from keras.activations import relu
from keras.initializers import RandomNormal
from keras.applications import *
import keras.backend as K
from pixel_shuffler import PixelShuffler

conv_init = RandomNormal(0, 0.02)

def conv_block(input_tensor, f):
    x = input_tensor
    x = Conv2D(f, kernel_size=3, strides=2, kernel_initializer=conv_init, use_bias=False, padding="same")(x)
    x = Activation("relu")(x)
    return x

def conv_block_d(input_tensor, f, use_instance_norm=True):
    x = input_tensor
    x = Conv2D(f, kernel_size=4, strides=2, kernel_initializer=conv_init, use_bias=False, padding="same")(x)
    x = LeakyReLU(alpha=0.2)(x)
    return x

def res_block(input_tensor, f):
    x = input_tensor
    x = Conv2D(f, kernel_size=3, kernel_initializer=conv_init, use_bias=False, padding="same")(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(f, kernel_size=3, kernel_initializer=conv_init, use_bias=False, padding="same")(x)
    x = add([x, input_tensor])
    x = LeakyReLU(alpha=0.2)(x)
    return x

def upscale_ps(filters, use_norm=True):
    def block(x):
        x = Conv2D(filters*4, kernel_size=3, use_bias=False, kernel_initializer=RandomNormal(0, 0.02), padding='same' )(x)
        x = LeakyReLU(0.1)(x)
        x = PixelShuffler()(x)
        return x
    return block

def Discriminator(nc_in, input_size=64):
    inp = Input(shape=(input_size, input_size, nc_in))
    #x = GaussianNoise(0.05)(inp)
    x = conv_block_d(inp, 64, False)
    x = conv_block_d(x, 128, False)
    x = conv_block_d(x, 256, False)
    out = Conv2D(1, kernel_size=4, kernel_initializer=conv_init, use_bias=False, padding="same", activation="sigmoid")(x)   
    return Model(inputs=[inp], outputs=out)

def Encoder(nc_in=3, input_size=64):
    inp = Input(shape=(input_size, input_size, nc_in))
    x = Conv2D(64, kernel_size=5, kernel_initializer=conv_init, use_bias=False, padding="same")(inp)
    x = conv_block(x,128)
    x = conv_block(x,256)
    x = conv_block(x,512) 
    x = conv_block(x,1024)
    x = Dense(1024)(Flatten()(x))
    x = Dense(4*4*1024)(x)
    x = Reshape((4, 4, 1024))(x)
    out = upscale_ps(512)(x)
    return Model(inputs=inp, outputs=out)

def Decoder_ps(nc_in=512, input_size=8):
    input_ = Input(shape=(input_size, input_size, nc_in))
    x = input_
    x = upscale_ps(256)(x)
    x = upscale_ps(128)(x)
    x = upscale_ps(64)(x)
    x = res_block(x, 64)
    x = res_block(x, 64)
    #x = Conv2D(4, kernel_size=5, padding='same')(x)   
    alpha = Conv2D(1, kernel_size=5, padding='same', activation="sigmoid")(x)
    rgb = Conv2D(3, kernel_size=5, padding='same', activation="tanh")(x)
    out = concatenate([alpha, rgb])
    return Model(input_, out )    