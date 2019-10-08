
from keras.models import Sequential, Model
from keras.layers import *
from keras.layers.advanced_activations import LeakyReLU
from keras.activations import relu
from keras.initializers import RandomNormal
from keras.applications import *
import keras.backend as K
from tensorflow.contrib.distributions import Beta
import tensorflow as tf
from keras.optimizers import Adam

from image_augmentation import random_transform
from image_augmentation import random_warp
from utils import get_image_paths, load_images, stack_images
from pixel_shuffler import PixelShuffler
from instance_normalization import InstanceNormalization

import time
import os
import numpy as np
from PIL import Image
import cv2
import glob
from random import randint, shuffle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import simplejson as json
from prefetch_generator import background
import socket

from umeyama import umeyama

from tensorflow.python.client import device_lib


import argparse
parser = argparse.ArgumentParser()
parser.add_argument( "--source-dir", type=str, dest="source_dir")
parser.add_argument( "--target-dir", type=str, dest="target_dir")
args = parser.parse_args()


from pathlib import Path

source_dir = Path(args.source_dir)
target_dir = Path(args.target_dir)

source_face_dir = source_dir / "extracted-faces"
target_face_dir = target_dir / "extracted-faces"

model_dir = target_dir / "models"
model_dir.mkdir(parents=True, exist_ok=True)

preview_dir = model_dir / "previews"
preview_dir.mkdir(parents=True, exist_ok=True)


from keras_vggface.vggface import VGGFace

base_network_PL = 'vgg16' # 'resnet50' or 'vgg16'

vggface = VGGFace(include_top=False, model=base_network_PL, input_shape=(224, 224, 3))

K.set_learning_phase(1)



channel_axis=-1
channel_first = False

"""
Base model:
1. "XGAN": standard autoencoder as generator + adversarial losses + encoding losses 
2. "VAE-GAN": variational autoencoder as generator + adversarial losses
3. "GAN": standard autoencoder as generator + adversarial losses
"""
base_model = "VAE-GAN"

"""
Decoder type:
1. Shared decoder
2. Separate decoders
"""
decoder_type = 1 if base_model == "VAE-GAN" else 2

IMAGE_SHAPE = (64, 64, 3)
nc_in = 3 # number of input channels of generators
nc_D_inp = 6 # number of input channels of discriminators
nz = 2048 if decoder_type == 1 else 1024 # nz=2048 according to VAE-GAN paper https://arxiv.org/abs/1512.09300
"""
Notes for latent dimension:
1. BicycleGAN use nz=8 for noise
2. From UNIT implementation: https://github.com/mingyuliutw/UNIT/blob/master/src/trainers/cocogan_nets_da.py
      self.g_vae = GaussianVAE2D(ch * 8, ch * 8, kernel_size=1, stride=1),
      where ch=32 by default. (notice that its 2D latent space, i.e., using Conv2D instead of Dense)
"""

use_perceptual_loss = True
use_lsgan = True
use_mixup = True # mixup paper: https://arxiv.org/abs/1710.09412
mixup_alpha = 0.2


small_batch_size = 16
standard_batch_size = 64
max_batch_size = 128

#small_batch_size = 4
#standard_batch_size = 8
#max_batch_size = 8

batchSize = small_batch_size


lrD = 1e-4 # Discriminator learning rate
lrG = 1e-4 # Generator learning rate

# Path of training images
img_dirA = str(source_face_dir) + '/*.*'
img_dirB = str(target_face_dir) + '/*.*'



conv_init = RandomNormal(0, 0.02)

def conv_block(input_tensor, f, k=3, strides=2, dilation_rate=1, use_inst_norm=True):
    x = input_tensor
    x = Conv2D(f, kernel_size=k, strides=strides, dilation_rate=dilation_rate, kernel_initializer=conv_init, use_bias=True, padding="same")(x)
    if use_inst_norm:
        x = InstanceNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    return x

def conv_block_d(input_tensor, f, use_inst_norm=True):
    x = input_tensor
    x = Conv2D(f, kernel_size=4, strides=2, kernel_initializer=conv_init, use_bias=True, padding="same")(x)
    if use_inst_norm:
        x = InstanceNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    return x

def res_block(input_tensor, f):
    x = input_tensor
    x = Conv2D(f, kernel_size=5, kernel_initializer=conv_init, use_bias=True, padding="same")(x)
    x = InstanceNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(f, kernel_size=5, kernel_initializer=conv_init, use_bias=True, padding="same")(x)
    x = InstanceNormalization()(x)
    x = add([x, input_tensor])
    return x

def upscale_ps(filters, dilation_rate=1, use_norm=True):
    def block(x):
        x = Conv2D(filters*4, kernel_size=3, dilation_rate=dilation_rate, use_bias=True, kernel_initializer=RandomNormal(0, 0.02), padding='same')(x)
        x = InstanceNormalization()(x)
        x = LeakyReLU(0.1)(x)
        x = PixelShuffler()(x)
        return x
    return block

def Discriminator(nc_in, input_size=64):
    inp = Input(shape=(input_size, input_size, nc_in))
    #x = GaussianNoise(0.05)(inp)
    x = conv_block_d(inp, 64, False)
    x = conv_block_d(x, 128)
    x = conv_block_d(x, 256)
    out = Conv2D(1, kernel_size=4, kernel_initializer=conv_init, use_bias=False, padding="same", activation="sigmoid")(x)   
    return Model(inputs=[inp], outputs=out)

def Discriminator2(nc_in, input_size=64):
    inp = Input(shape=(input_size, input_size, nc_in))
    x = Conv2D(32, kernel_size=5, kernel_initializer=conv_init, use_bias=False, padding="same")(inp)
    x = LeakyReLU(0.1)(x)
    #x = GaussianNoise(0.05)(inp)
    x = conv_block_d(x, 64, False)
    x = conv_block_d(x, 128)
    out0 = x
    x = conv_block_d(x, 256)
    out1 = x
    x = conv_block_d(x, 512)
    out2 = x
    out3 = Conv2D(1, kernel_size=3, kernel_initializer=conv_init, use_bias=False, padding="same", activation="sigmoid")(x)   
    return Model(inputs=[inp], outputs=[out0, out1, out2, out3]) 

def Discriminator_code():
    inp = Input(shape=(1024, ))
    x = Dense(256)(inp)
    x = InstanceNormalization()(x)
    x = Activation('relu')(x)
    x = Dense(128)(x)
    x = InstanceNormalization()(x)
    x = Activation('relu')(x)
    x = Dense(32)(x)
    x = InstanceNormalization()(x)
    x = Activation('relu')(x)
    out = Dense(1, activation='sigmoid')(x)   
    return Model(inputs=[inp], outputs=out)

def Encoder(nc_in=3, input_size=64):
    def l2_norm(x):
        epsilon = 1e-12
        x_norm = K.sqrt(K.sum(K.square(x)))
        return x / (x_norm + epsilon)  
  
    inp = Input(shape=(input_size, input_size, nc_in))
    x = Conv2D(64, kernel_size=5, kernel_initializer=conv_init, use_bias=False, padding="same")(inp)
    x = LeakyReLU(0.1)(x)
    x = conv_block(x,128)
    x = conv_block(x,256)
    x = conv_block(x,512)
    x = conv_block(x,1024)
    x = Dense(1024)(Flatten()(x))
    x = InstanceNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = Dense(1024)(x)
    x = Lambda(l2_norm)(x)
    code = x
    x = Dense(4*4*1024)(x)
    x = Reshape((4, 4, 1024))(x)
    out = upscale_ps(512, dilation_rate=1)(x)
    return Model(inputs=inp, outputs=[out, code])

def Encoder2(nc_in=3, input_size=64):
    def l2_norm(x):
        epsilon = 1e-12
        x_norm = K.sqrt(K.sum(K.square(x)))
        return x / (x_norm + epsilon)  
  
    inp = Input(shape=(input_size, input_size, nc_in))
    x = Conv2D(64, kernel_size=5, kernel_initializer=conv_init, use_bias=False, padding="same")(inp)
    x = LeakyReLU(0.1)(x)
    x = conv_block(x,128)
    x = conv_block(x,256)
    x = conv_block(x,512)
    x = conv_block(x,1024)
    #x = Flatten()(x)
    out = x
    return Model(inputs=inp, outputs=out)

def Encoder_vae(nc_in=3, input_size=64):  
    inp = Input(shape=(input_size, input_size, nc_in))
    x = Conv2D(64, kernel_size=5, kernel_initializer=conv_init, use_bias=False, padding="same")(inp)
    x = LeakyReLU(0.1)(x)
    x = conv_block(x,128)
    x = conv_block(x,256)
    x = conv_block(x,512)
    x = conv_block(x,1024)
    out = x
    return Model(inputs=inp, outputs=out)

def Decoder_ps(nc_in=512, input_size=8):
    inp = Input(shape=(input_size, input_size, nc_in))
    code = Input(shape=(1024,))
    
    x_code = Dense(4*4*512)(code)
    x_code = Reshape((4, 4, 512))(x_code)
    x_code = upscale_ps(256)(x_code)
    x_code = upscale_ps(128)(x_code)
    
    x = inp
    x = upscale_ps(256)(x)
    x = concatenate([x, x_code])
    x = upscale_ps(128)(x)
    x = upscale_ps(64)(x)
    x = res_block(x, 64)
    x = res_block(x, 64)
    alpha = Conv2D(1, kernel_size=5, padding='same', activation="sigmoid")(x)
    bgr = Conv2D(3, kernel_size=5, padding='same', activation="tanh")(x)
    out = concatenate([alpha, bgr])
    return Model([inp, code], [out, code])  

def Decoder_code(nc_in=1024, input_size=4):        
    inp = Input(shape=(input_size, input_size, nc_in))
    
    x = inp
    x = Flatten()(x)
    x_code = Dense(1024)(x)
    x_code = InstanceNormalization()(x_code)
    x_code = LeakyReLU(0.1)(x_code)
    x_code = Dense(1024)(x_code)
    x_code = Dense(4*4*512)(x_code)
    x_code = Reshape((4, 4, 512))(x_code)
    out = x_code
    return Model(inp, out)   
  
def Decoder_common(nc_in=1024, input_size=4):
    inp = Input(shape=(input_size, input_size, nc_in))
    
    x = inp
    x = upscale_ps(512)(x)
    x = upscale_ps(256)(x)
    x = upscale_ps(128)(x)
    x = upscale_ps(64)(x)
    x = res_block(x, 64)
    x = res_block(x, 64)
    alpha = Conv2D(1, kernel_size=5, padding='same', activation="sigmoid")(x)
    bgr = Conv2D(3, kernel_size=5, padding='same', activation="tanh")(x)
    out = concatenate([alpha, bgr])
    return Model(inp, out) 

def Decoder_code_vae(nc_in=1024, input_size=4, z_dim=32):   
    """VAE implementation reference: https://github.com/EmilienDupont/vae-concrete """
    def l2_norm(x):
        epsilon = 1e-12
        x_norm = K.sqrt(K.sum(K.square(x)))
        return x / (x_norm + epsilon)  
    def sampling_normal(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(z_dim,), mean=0., stddev=1.)
        return z_mean + K.exp(z_log_var / 2) * epsilon
      
    inp = Input(shape=(input_size, input_size, nc_in))
    
    x = inp
    x = Flatten()(x)
    x_code = Dense(1024)(x)
    x_code = InstanceNormalization()(x_code)
    x_code = LeakyReLU(0.1)(x_code)
    x_code = Dense(1024)(x_code)    
    x_code = InstanceNormalization()(x_code)
    x_code = LeakyReLU(0.1)(x_code)    
    z_mean = Dense(z_dim)(x_code) # latent dim = 32
    z_log_var = Dense(z_dim)(x_code)    
    encoding = Lambda(sampling_normal)([z_mean, z_log_var])    
    out = encoding
    return Model(inp, [out, z_mean, z_log_var])  

def Decoder_code_soft_gate(nc_in=1024, input_size=4, nc_out=512):   
    def l2_norm(x):
        epsilon = 1e-12
        x_norm = K.sqrt(K.sum(K.square(x)))
        return x / (x_norm + epsilon)  
      
    inp = Input(shape=(input_size, input_size, nc_in))
    
    x = inp
    x = Flatten()(x)
    x_code = Dense(1024)(x)
    x_code = InstanceNormalization()(x_code)
    x_code = LeakyReLU(0.1)(x_code)
    x_code = Dense(1024)(x_code)    
    soft_gate = InstanceNormalization()(x_code)
    soft_gate = LeakyReLU(0.1)(soft_gate)
    soft_gate = Dense(1024)(soft_gate)  
    soft_gate = InstanceNormalization()(soft_gate)
    soft_gate = LeakyReLU(0.1)(soft_gate)
    soft_gate = Dense(nc_out, activation='sigmoid')(soft_gate)    
    x_code = Lambda(l2_norm)(x_code)
    x_code = Dense(4*4*nc_out)(x_code)
    x_code = Reshape((4, 4, nc_out))(x_code)    
    out = multiply([soft_gate, x_code])    
    return Model(inp, out)  
  
def Decoder_common_vae(nc_in=64, input_size=4):
    inp = Input(shape=(nc_in, ))
    
    x = inp    
    x = Dense(4*4*1024)(x)
    x = InstanceNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = Reshape((4, 4, 1024))(x)
    x = upscale_ps(512)(x)
    x = upscale_ps(256)(x)
    x = upscale_ps(128)(x)
    x = upscale_ps(64)(x)
    x = res_block(x, 64)
    x = res_block(x, 64)
    alpha = Conv2D(1, kernel_size=5, padding='same', activation="sigmoid")(x)
    bgr = Conv2D(3, kernel_size=5, padding='same', activation="tanh")(x)
    out = concatenate([alpha, bgr])
    return Model(inp, out)


# In[ ]:


def get_generator_vae(x, y, encoder, decoder_code1, decoder_code2, decoder_common):
    enc_x = encoder(x)
    enc_y = encoder(y)
    latent_z1, z1_mean, z1_log_var = decoder_code1(enc_x)
    latent_z2, z2_mean, z2_log_var = decoder_code2(enc_y)
    concat_latent_zs = concatenate([latent_z1, latent_z2])
    out = decoder_common(concat_latent_zs)
    netG = Model([x, y], [out, z1_mean, z1_log_var, z2_mean, z2_log_var])
    return netG  

if base_model == "XGAN":    
    encoder = Encoder()
    decoder_A = Decoder_ps()
    decoder_B = Decoder_ps()
    x = Input(shape=IMAGE_SHAPE)
    netGA = Model(x, decoder_A(encoder(x)))
    netGB = Model(x, decoder_B(encoder(x)))
    
elif base_model == "VAE-GAN":
    encoder = Encoder_vae()
    decoder_A = Decoder_code_vae(z_dim=nz//2)
    decoder_B = Decoder_code_vae(z_dim=nz//2)
    # Since VAE has a more representative latent space, 
    # we introduce bottlneck to AnB decoder in hope of it will encode the most important facial attributes.
    # And we can perhaps manipulate these attributes in the future.
    # Result: Still under experiment (with beta=0.01).
    decoder_AnB = Decoder_code_vae(z_dim=nz//2) # z_dim=nz//64?
    
    x = Input(shape=IMAGE_SHAPE)
    y = Input(shape=IMAGE_SHAPE)
    
    if decoder_type == 1:
        decoder_common = Decoder_common_vae(nc_in=(nz//2 + nz//2))
        netGA = get_generator_vae(x, y, encoder, decoder_AnB, decoder_A, decoder_common)
        netGB = get_generator_vae(x, y, encoder, decoder_AnB, decoder_B, decoder_common)
    elif decoder_type == 2:
        decoder_A2 = Decoder_common_vae(nc_in=nz)
        decoder_B2 = Decoder_common_vae(nc_in=nz)
        netGA = get_generator_vae(x, y, encoder, decoder_AnB, decoder_A, decoder_A2)
        netGB = get_generator_vae(x, y, encoder, decoder_AnB, decoder_B, decoder_B2)
        
elif base_model == "GAN":
    encoder = Encoder2()
    decoder_A = Decoder_code()
    decoder_B = Decoder_code()
    decoder_AnB = Decoder_code()

    x = Input(shape=IMAGE_SHAPE)
    y = Input(shape=IMAGE_SHAPE)

    if decoder_type == 1:
        decoder_common = Decoder_common()
        netGA = Model([x, y], decoder_common(concatenate([decoder_AnB(encoder(x)), decoder_A(encoder(y))])))
        netGB = Model([x, y], decoder_common(concatenate([decoder_AnB(encoder(x)), decoder_B(encoder(y))])))
    elif decoder_type == 2:
        decoder_A2 = Decoder_common()
        decoder_B2 = Decoder_common()
        netGA = Model([x, y], decoder_A2(concatenate([decoder_AnB(encoder(x)), decoder_A(encoder(y))])))
        netGB = Model([x, y], decoder_B2(concatenate([decoder_AnB(encoder(x)), decoder_B(encoder(y))])))


netDA = Discriminator(nc_D_inp)
netDB = Discriminator(nc_D_inp)
netDA2 = Discriminator2(nc_D_inp//2)
netDB2 = Discriminator2(nc_D_inp//2)

if base_model == "XGAN":
    netD_code = Discriminator_code()


try:
    print("Loading model")
    encoder.load_weights(str(model_dir / "encoder.h5"))
    decoder_A.load_weights(str(model_dir / "decoder_A.h5"))
    decoder_B.load_weights(str(model_dir / "decoder_B.h5"))
    if base_model == "VAE-GAN" or base_model == "GAN":
        decoder_AnB.load_weights(str(model_dir / "decoder_AnB.h5"))
        if decoder_type == 1:
            decoder_common.load_weights(str(model_dir / "decoder_common.h5"))
        elif decoder_type == 2:
            decoder_A2.load_weights(str(model_dir / "decoder_A2.h5"))
            decoder_B2.load_weights(str(model_dir / "decoder_B2.h5"))
    netDA.load_weights(str(model_dir / "netDA.h5") )
    netDB.load_weights(str(model_dir / "netDB.h5") )
    netDA2.load_weights(str(model_dir / "netDA2.h5") )
    netDB2.load_weights(str(model_dir / "netDB2.h5") )
    if base_model == "XGAN":
        netD_code.load_weights(str(model_dir / "netD_code.h5"))
    print ("model loaded.")
except:
    print ("Failed loading. Something goes wrong with weights files.")
    pass


def save_progress(thedir, progress):
    print("Saving progress")
    progress_file = Path(thedir / 'progress.json')
    with progress_file.open('w') as f:
        j = json.dumps(progress, ensure_ascii=False )
        f.write(j)


def load_progress(thedir):
    try:
        progress = Path(thedir / 'progress.json')
        with progress.open() as f:
            progress = json.load(f)

        return progress
    except Exception as e:
        print("No progress data loaded.")
        print(e)
        p=dict(epoch=list(),
                iterations=list(),
                errDA=list(),
                errDB=list(),
                errDA2=list(),
                errDB2=list(),
                errGA=list(),
                errGB=list())
        return p

def plot_progress(thedir, progress):
  print("Plot progress.")
  fig = plt.figure(figsize=(20,10))
  plt.cla()
  ax1 = fig.gca()
  ax2 = ax1.twinx()

  ax1.plot(progress['errDA'], color='lightblue')
  ax1.plot(progress['errDB'], color='lightgreen')

  if 'errDA2' in progress:
    ax1.plot(progress['errDA2'], color='lightblue')

  if 'errDB2' in progress:
    ax1.plot(progress['errDB2'], color='lightgreen')

  ax2.plot(progress['errGA'], color='blue')
  ax2.plot(progress['errGB'], color='green')
  
  plt.savefig(str(thedir / 'progress.png'))
  plt.clf()
  plt.close()
  print("Plot done")

def save_models(thedir):
    print("Saving models")
    global encoder, decoder_A, decoder_B, decoder_AnB, decoder_type, decoder_common, decoder_A2, decoder_B2, netD_code, base_model
    encoder.save_weights(str(thedir / "encoder.h5"))
    decoder_A.save_weights(str(thedir / "decoder_A.h5"))
    decoder_B.save_weights(str(thedir / "decoder_B.h5")  )
    if base_model == "VAE-GAN" or base_model == "GAN":
        decoder_AnB.save_weights(str(thedir / "decoder_AnB.h5")     )
        if decoder_type == 1:
            decoder_common.save_weights(str(thedir / "decoder_common.h5"))
        elif decoder_type == 2:
            decoder_A2.save_weights(str(thedir / "decoder_A2.h5"))
            decoder_B2.save_weights(str(thedir / "decoder_B2.h5"))
    netDA.save_weights(str(thedir / "netDA.h5"))
    netDB.save_weights(str(thedir / "netDB.h5"))
    netDA2.save_weights(str(thedir / "netDA2.h5"))
    netDB2.save_weights(str(thedir / "netDB2.h5"))
    if base_model == "XGAN":
        netD_code.save_weights("models/netD_code.h5")


if base_model == "XGAN":
    def cycle_variables(netG):
        distorted_input = netG.inputs[0]
        fake_output = netG.outputs[0]
        code = netG.outputs[1]
        
        alpha = Lambda(lambda x: x[:,:,:, :1])(fake_output)
        bgr = Lambda(lambda x: x[:,:,:, 1:])(fake_output)
        masked_fake_output = alpha * bgr + (1-alpha) * distorted_input 

        fn_generate = K.function([distorted_input], [masked_fake_output])
        fn_mask = K.function([distorted_input], [concatenate([alpha, alpha, alpha])])
        fn_abgr = K.function([distorted_input], [concatenate([alpha, bgr])])
        fn_bgr = K.function([distorted_input], [bgr])
        return distorted_input, fake_output, code, alpha, fn_generate, fn_mask, fn_abgr, fn_bgr
        
elif base_model == "VAE-GAN":
    def cycle_variables(netG, decoder_common=None):      
        distorted_input_x = netG.inputs[0]
        distorted_input_y = netG.inputs[1]
        fake_output = netG.outputs[0]
        z1_mean, z1_log_var = netG.outputs[1], netG.outputs[2]
        z2_mean, z2_log_var = netG.outputs[3], netG.outputs[4]
        zs = [z1_mean, z1_log_var, z2_mean, z2_log_var]
        
        concat_latent_z_means = concatenate([z1_mean, z2_mean]) # latent code w/o var. noise
        fake_output_inference = decoder_common(concat_latent_z_means)
        
        alpha = Lambda(lambda x: x[:,:,:, :1])(fake_output)
        alpha_inference = Lambda(lambda x: x[:,:,:, :1])(fake_output_inference)
        bgr_inference = Lambda(lambda x: x[:,:,:, 1:])(fake_output_inference)        
        masked_fake_output = alpha_inference * bgr_inference + (1-alpha_inference) * distorted_input_x 
    
        fn_generate = K.function([distorted_input_x, distorted_input_y], [masked_fake_output])
        fn_mask = K.function([distorted_input_x, distorted_input_y], [concatenate([alpha_inference, alpha_inference, alpha_inference])])
        fn_bgr = K.function([distorted_input_x, distorted_input_y], [bgr_inference])
        return distorted_input_x, distorted_input_y, fake_output, alpha, zs, fn_generate, fn_mask, fn_bgr
        
elif base_model == "GAN":
    def cycle_variables(netG):
        distorted_input_x = netG.inputs[0]
        distorted_input_y = netG.inputs[1]
        fake_output = netG.outputs[0]
        
        alpha = Lambda(lambda x: x[:,:,:, :1])(fake_output)
        bgr = Lambda(lambda x: x[:,:,:, 1:])(fake_output)
        masked_fake_output = alpha * bgr + (1-alpha) * distorted_input_x 

        fn_generate = K.function([distorted_input_x, distorted_input_y], [masked_fake_output])
        fn_mask = K.function([distorted_input_x, distorted_input_y], [concatenate([alpha, alpha, alpha])])
        fn_abgr = K.function([distorted_input_x, distorted_input_y], [concatenate([alpha, bgr])])
        fn_bgr = K.function([distorted_input_x, distorted_input_y], [bgr])
        return distorted_input_x, distorted_input_y, fake_output, alpha, fn_generate, fn_mask, fn_abgr, fn_bgr


if base_model == "XGAN":
    distorted_A, fake_A, code_A, mask_A, path_A, path_mask_A, path_abgr_A, path_bgr_A = cycle_variables(netGA)
    distorted_B, fake_B, code_B, mask_B, path_B, path_mask_B, path_abgr_B, path_bgr_B = cycle_variables(netGB)
    real_A = Input(shape=IMAGE_SHAPE)
    real_B = Input(shape=IMAGE_SHAPE)
    
elif base_model == "VAE-GAN":
    if decoder_type == 1:
        distorted_x_A, distorted_y_A, fake_A, mask_A, zs_A, path_A, path_mask_A, path_bgr_A = cycle_variables(netGA, decoder_common)
        distorted_x_B, distorted_y_B, fake_B, mask_B, zs_B, path_B, path_mask_B, path_bgr_B = cycle_variables(netGB, decoder_common)
    elif decoder_type == 2:
        distorted_x_A, distorted_y_A, fake_A, mask_A, zs_A, path_A, path_mask_A, path_bgr_A = cycle_variables(netGA, decoder_A2)
        distorted_x_B, distorted_y_B, fake_B, mask_B, zs_B, path_B, path_mask_B, path_bgr_B = cycle_variables(netGB, decoder_B2)
    real_A = Input(shape=IMAGE_SHAPE)
    real_B = Input(shape=IMAGE_SHAPE)
    
elif base_model == "GAN":
    distorted_x_A, distorted_y_A, fake_A, mask_A, path_A, path_mask_A, path_abgr_A, path_bgr_A = cycle_variables(netGA)
    distorted_x_B, distorted_y_B, fake_B, mask_B, path_B, path_mask_B, path_abgr_B, path_bgr_B = cycle_variables(netGB)
    real_A = Input(shape=IMAGE_SHAPE)
    real_B = Input(shape=IMAGE_SHAPE)

# # 8. Loss Functions
# Loss weighting for generators
w_D1 = .5
w_D2 = .5
w_adv_PL = 1.
w_recon_loss = 3.
w_kl_loss = 0.01 # The hyperparameter beta of beta-VAE https://openreview.net/forum?id=Sy2fzU9gl
w_pl_vgg = (0.01, 0.003, 0.003, 0.001, 0.01)
w_mask_fo = 0.1
w_mask = 0.1
"""
Notes for beta (w_kl_loss):
beta-VAE suggests using beta > 1 (e.g., 5, 20, 250) to disentangle latent features.
However, the KL loss in such case will be much larger than other losses.
Also we don't have much interest in disentangled features (at least for now).
Thus beta is set to 0.01 (same with BicycleGAN default value).
"""

# Loss weighting for discriminators
m_adv_PL = 0.65 # heuristically chosing a value between 0.5 and 1: 0.5 + (1-0.5)/3


if use_lsgan:
    loss_fn = lambda output, target : K.mean(K.abs(K.square(output-target)))
else:
    loss_fn = lambda output, target : -K.mean(K.log(output+1e-12)*target+K.log(1-output+1e-12)*(1-target))
    
loss_fn_bce = lambda output, target : -K.mean(K.log(output+1e-12)*target+K.log(1-output+1e-12)*(1-target))



# https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cosine.html
def cos_distance(x1, x2):
    x1 = K.l2_normalize(x1, axis=-1)
    x2 = K.l2_normalize(x2, axis=-1)
    return K.mean(1 - K.sum((x1 * x2), axis=-1))

def first_order(x, axis=1):
    img_nrows = x.shape[1]
    img_ncols = x.shape[2]
    if axis == 1:
        return K.abs(x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, 1:, :img_ncols - 1, :])
    elif axis == 2:
        return K.abs(x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, :img_nrows - 1, 1:, :])
    else:
        return None   

# https://github.com/EmilienDupont/vae-concrete/blob/master/util.py
def kl_normal(z_mean, z_log_var):
    kl_per_example = .5 * (K.sum(K.square(z_mean) + K.exp(z_log_var) - 1 - z_log_var, axis=1))
    return K.mean(kl_per_example)


# ========== Define Perceptual Loss Model==========
if use_perceptual_loss:
    print ("Using perceptual loss.")
    vggface.trainable = False
    if base_network_PL == "resnet50":
        # ResNet50 feats
        out_size55 = vggface.layers[36].output
        out_size28 = vggface.layers[78].output
        out_size7 = vggface.layers[-2].output
        vggface_feat = Model(vggface.input, [out_size55, out_size28, out_size7])
    elif base_network_PL == "vgg16":
        # VGG16 feats
        out_size224 = vggface.layers[1].output
        out_size112 = vggface.layers[4].output
        out_size56 = vggface.layers[7].output
        out_size28 = vggface.layers[11].output
        out_size14 = vggface.layers[15].output
        vggface_feat = Model(vggface.input, [out_size224, out_size112, out_size56, out_size28, out_size14])
    vggface_feat.trainable = False
else:
    print ("Not using perceptual loss.")
    vggface_feat = None


def define_loss(netD, netD2, netG, real, fake_abgr, distorted, zs=None, vggface_feat=None, domain="A", netD_code=None, real_code=None):
    alpha = Lambda(lambda x: x[:,:,:, :1])(fake_abgr)
    fake_bgr = Lambda(lambda x: x[:,:,:, 1:])(fake_abgr)
    fake = alpha * fake_bgr + (1-alpha) * distorted
    
    if use_mixup:
        dist = Beta(mixup_alpha, mixup_alpha)
        lam = dist.sample()
        
        mixup = lam * concatenate([real, distorted]) + (1 - lam) * concatenate([fake, distorted])
        out_D1_mixup = netD(mixup)
        # GAN loss1
        loss_D = loss_fn(out_D1_mixup, lam * K.ones_like(out_D1_mixup)) 
        loss_G = w_D1 * loss_fn(out_D1_mixup, (1 - lam) * K.ones_like(out_D1_mixup))
        
        lam2 = dist.sample()
        mixup2 = lam2 * real + (1 - lam2) * fake_bgr
        out3_D2_mixup = netD2(mixup2)[3]  
        # GAN loss2
        loss_D2 = loss_fn(out3_D2_mixup, lam2 * K.ones_like(out3_D2_mixup))
        loss_G += w_D2 * loss_fn(out3_D2_mixup, (1 - lam) * K.ones_like(out3_D2_mixup))    
        # Perceptual adversarial loss
        out0_D2_real, out1_D2_real, out2_D2_real, _ = netD2(real)
        out0_D2_fake, out1_D2_fake, out2_D2_fake, _ = netD2(fake_bgr)
        adversarial_perceptual_loss_D2 = K.mean(K.abs(out0_D2_fake - out0_D2_real)) + K.mean(K.abs(out1_D2_fake - out1_D2_real)) + K.mean(K.abs(out2_D2_fake - out2_D2_real))
        loss_G += w_adv_PL * adversarial_perceptual_loss_D2
        loss_D2 += K.maximum(K.zeros_like(adversarial_perceptual_loss_D2), m_adv_PL - adversarial_perceptual_loss_D2)
        
        if base_model == "XGAN":
            rec_code = netG([fake_bgr])[1]
            output_real_code = netD_code([real_code])
            # Target of domain A = 1, domain B = 0
            if domain == "A":
                loss_D_code = loss_fn_bce(output_real_code, K.ones_like(output_real_code))
                loss_G += .1 * loss_fn(output_real_code, K.zeros_like(output_real_code))
            elif domain == "B":
                loss_D_code = loss_fn_bce(output_real_code, K.zeros_like(output_real_code))
                loss_G += .1 * loss_fn(output_real_code, K.ones_like(output_real_code))
            loss_G += 1. * cos_distance(rec_code, real_code)
    
    else:
        print ("use_mixup=False is not supported!") 
        
    # Reconstruction loss. L1 distance between reconstructed image and ground truth image. 
    loss_G += w_recon_loss * K.mean(K.abs(fake_bgr - real)) 
    
    if base_model == "VAE-GAN":
        z1_mean, z1_log_var, z2_mean, z2_log_var = zs
        # VAE loss2: KL divergence between N(0,1) and N(z_mean, exp(z_log_var)).    
        loss_G += w_kl_loss * kl_normal(z1_mean, z1_log_var) / 2 # Dividing by 2 because AnB KL loss will be updated twice within one iter.
        loss_G += w_kl_loss * kl_normal(z2_mean, z2_log_var)
    
    # Perceptual Loss
    if not vggface_feat is None:
        def preprocess_vggface(x):
            if base_network_PL == "resnet50":
                x = (x + 1)/2 * 255 # channel order: BGR
                x -= [91.4953, 103.8827, 131.0912]
            elif base_network_PL == "vgg16":
                x = (x + 1)/2 * 255 # channel order: BGR
                x -= [93.5940, 104.7624, 129.1863]
            return x
        real_sz224 = tf.image.resize_images(real, [224, 224])
        real_sz224 = Lambda(preprocess_vggface)(real_sz224)
        fake_sz224 = tf.image.resize_images(fake, [224, 224]) 
        fake_sz224 = Lambda(preprocess_vggface)(fake_sz224)
        
        # ResNet50 PL
        #pl_params = (0.03, 0.2, 0.3)
        #real_feat55, real_feat28, real_feat7 = vggface_feat(real_sz224)
        #fake_feat55, fake_feat28, fake_feat7  = vggface_feat(fake_sz224)    
        #loss_G += pl_params[0] * K.mean(K.square(fake_feat7 - real_feat7))
        #loss_G += pl_params[1] * K.mean(K.abs(fake_feat28 - real_feat28))
        #loss_G += pl_params[2] * K.mean(K.abs(fake_feat55 - real_feat55))
        
        # VGG16 PL
        pl_params = w_pl_vgg
        real_feat224, real_feat112, real_feat56, real_feat28, real_feat14 = vggface_feat(real_sz224)
        fake_feat224, fake_feat112, fake_feat56, fake_feat28, fake_feat14  = vggface_feat(fake_sz224)    
        loss_G += pl_params[0] * K.mean(K.abs(fake_feat14 - real_feat14))
        loss_G += pl_params[1] * K.mean(K.abs(fake_feat28 - real_feat28))
        loss_G += pl_params[2] * K.mean(K.abs(fake_feat56 - real_feat56))
        loss_G += pl_params[3] * K.mean(K.abs(fake_feat112 - real_feat112)) 
        loss_G += pl_params[4] * K.mean(K.abs(fake_feat224 - real_feat224))   
    if base_model =="GAN" or base_model =="VAE-GAN":
        return loss_D, loss_D2, loss_G
    elif base_model == "XGAN":
        return loss_D, loss_D2, loss_G, loss_D_code


def build_training_functions(use_PL=False, use_mask_hinge_loss=False, m_mask=0.5):
    print("Building training functions")

    global netGA, netDA, real_A, fake_A, distorted_A, mask_A
    global netGB, netDB, real_B, fake_B, distorted_B, mask_B
    global netDA_train, netDA2_train, netGA_train, netDB_train, netDB2_train, netGB_train
    global vggface_feat
    global w_mask, w_mask_fo

    if use_PL:
        vggface_feat_local = vggface_feat
    else:
        vggface_feat_local = None

    if base_model == "XGAN":
        loss_DA, loss_DA2, loss_GA, loss_DA_code = define_loss(netDA, netDA2, netGA, real_A, fake_A, distorted_A, 
                                                               zs=None, vggface_feat=vggface_feat_local, 
                                                               domain="A", netD_code=netD_code, real_code=code_A)
        loss_DB, loss_DB2, loss_GB, loss_DB_code = define_loss(netDB, netDB2, netGB, real_B, fake_B, distorted_B, 
                                                               zs=None, vggface_feat=vggface_feat_local, 
                                                               domain="B", netD_code=netD_code, real_code=code_B)
    elif base_model == "VAE-GAN":  
        loss_DA, loss_DA2, loss_GA = define_loss(netDA, netDA2, netGA, real_A, fake_A, 
                                                 (distorted_x_A+distorted_y_A)/2, 
                                                 zs=zs_A, vggface_feat=vggface_feat_local)
        loss_DB, loss_DB2, loss_GB = define_loss(netDB, netDB2, netGB, real_B, fake_B, 
                                                 (distorted_x_B+distorted_y_B)/2, 
                                                 zs=zs_B, vggface_feat=vggface_feat)
    elif base_model == "GAN":
        loss_DA, loss_DA2, loss_GA = define_loss(netDA, netDA2, netGA, real_A, fake_A, 
                                                 (distorted_x_A+distorted_y_A)/2, 
                                                 zs=None, vggface_feat=vggface_feat_local)
        loss_DB, loss_DB2, loss_GB = define_loss(netDB, netDB2, netGB, real_B, fake_B, 
                                                 (distorted_x_B+distorted_y_B)/2, 
                                                 zs=None, vggface_feat=vggface_feat_local)

    # Alpha mask loss
    if not use_mask_hinge_loss:
        loss_GA += 1e-2 * K.mean(K.abs(mask_A))
        loss_GB += 1e-2 * K.mean(K.abs(mask_B))
    else:
        loss_GA += w_mask * K.mean(K.maximum(0., m_mask - mask_A))
        loss_GB += w_mask * K.mean(K.maximum(0., m_mask - mask_B))

    loss_GA += w_mask_fo * K.mean(first_order(mask_A, axis=1))
    loss_GA += w_mask_fo * K.mean(first_order(mask_A, axis=2))
    loss_GB += w_mask_fo * K.mean(first_order(mask_B, axis=1))
    loss_GB += w_mask_fo * K.mean(first_order(mask_B, axis=2))


    weightsDA = netDA.trainable_weights
    weightsDA2 = netDA2.trainable_weights
    weightsGA = netGA.trainable_weights
    weightsDB = netDB.trainable_weights
    weightsDB2 = netDB2.trainable_weights
    weightsGB = netGB.trainable_weights


    # Adam(..).get_updates(...)

    """
    # Using the following update function spped up training time (per iter.) by ~15%.
    training_updates = Adam(lr=lrD, beta_1=0.5).get_updates(weightsDA+weightsDA2,[],loss_DA+loss_DA2)
    netDA_train = K.function([distorted_A, real_A],[loss_DA+loss_DA2], training_updates)
    """
    if base_model == "VAE-GAN" or base_model == "GAN":
        training_updates = Adam(lr=lrD, beta_1=0.5).get_updates(weightsDA,[],loss_DA)
        netDA_train = K.function([distorted_x_A, distorted_y_A, real_A],[loss_DA], training_updates)
        training_updates = Adam(lr=lrD, beta_1=0.5).get_updates(weightsDA2,[],loss_DA2)
        netDA2_train = K.function([distorted_x_A, distorted_y_A, real_A],[loss_DA2], training_updates)
        training_updates = Adam(lr=lrG, beta_1=0.5).get_updates(weightsGA,[], loss_GA)
        netGA_train = K.function([distorted_x_A, distorted_y_A, real_A], [loss_GA], training_updates)

        training_updates = Adam(lr=lrD, beta_1=0.5).get_updates(weightsDB,[],loss_DB)
        netDB_train = K.function([distorted_x_B, distorted_y_B, real_B],[loss_DB], training_updates)
        training_updates = Adam(lr=lrD, beta_1=0.5).get_updates(weightsDB2,[],loss_DB2)
        netDB2_train = K.function([distorted_x_B, distorted_y_B, real_B],[loss_DB2], training_updates)
        training_updates = Adam(lr=lrG, beta_1=0.5).get_updates(weightsGB,[], loss_GB)
        netGB_train = K.function([distorted_x_B, distorted_y_B, real_B], [loss_GB], training_updates)
        
    elif base_model =="XGAN":
        training_updates = Adam(lr=lrD, beta_1=0.5).get_updates(weightsDA,[],loss_DA)
        netDA_train = K.function([distorted_A, real_A],[loss_DA], training_updates)
        training_updates = Adam(lr=lrD, beta_1=0.5).get_updates(weightsDA2,[],loss_DA2)
        netDA2_train = K.function([distorted_A, real_A],[loss_DA2], training_updates)
        training_updates = Adam(lr=lrG, beta_1=0.5).get_updates(weightsGA,[], loss_GA)
        netGA_train = K.function([distorted_A, real_A], [loss_GA], training_updates)

        training_updates = Adam(lr=lrD, beta_1=0.5).get_updates(weightsDB,[],loss_DB)
        netDB_train = K.function([distorted_B, real_B],[loss_DB], training_updates)
        training_updates = Adam(lr=lrD, beta_1=0.5).get_updates(weightsDB2,[],loss_DB2)
        netDB2_train = K.function([distorted_B, real_B],[loss_DB2], training_updates)
        training_updates = Adam(lr=lrG, beta_1=0.5).get_updates(weightsGB,[], loss_GB)
        netGB_train = K.function([distorted_B, real_B], [loss_GB], training_updates)
      
        weightsD_code = netD_code.trainable_weights
        training_updates = Adam(lr=lrD, beta_1=0.5).get_updates(weightsD_code,[], loss_DA_code)
        netDA_code_train = K.function([distorted_A, real_A],[loss_DA_code], training_updates)
        training_updates = Adam(lr=lrD, beta_1=0.5).get_updates(weightsD_code,[], loss_DB_code)
        netDB_code_train = K.function([distorted_B, real_B],[loss_DB_code], training_updates)

    print ("Loss configuration:")
    print ("use_PL = " + str(use_PL))
    print ("use_mask_hinge_loss = " + str(use_mask_hinge_loss))
    print ("m_mask = " + str(m_mask))


def set_batch_size(bs):
    global batchSize
    
    global train_batchA, train_batchB, train_A, train_B
    global epoch, warped_A, target_A, warped_B, target_B
    batchSize = bs
    train_batchA = minibatch(train_A, batchSize)
    train_batchB = minibatch(train_B, batchSize)
    epoch, warped_A, target_A = next(train_batchA) 
    epoch, warped_B, target_B = next(train_batchB) 
    print("New batch size: {:d}".format(batchSize))


schedule_step = -1 # 
def schedule_training_functions(gen_iterations, display_iters, TOTAL_ITERS):
    global schedule_step
    global batchSize
  # Loss function automation
    global max_batch_size, standard_batch_size, small_batch_size

    if gen_iterations < 0.1 * TOTAL_ITERS:
        if schedule_step == 0:
            return
        schedule_step = 0
        #start with a small batch size to quickly eat up most of the loss
        set_batch_size(small_batch_size)

        loss_config = dict(use_PL=False, 
                           use_mask_hinge_loss=False
                           )

        build_training_functions(**loss_config)


    elif gen_iterations < 0.2 * TOTAL_ITERS:
       
        if schedule_step == 1:
            return
        schedule_step = 1
        set_batch_size(standard_batch_size)
        
        loss_config = dict(use_PL=False, 
                           use_mask_hinge_loss=False
                           )

        build_training_functions(**loss_config)

    elif gen_iterations < 0.3 * TOTAL_ITERS:

        if schedule_step == 2:
            return
        schedule_step = 2

        set_batch_size(standard_batch_size)        
        loss_config = dict(
                use_PL=True,
                use_mask_hinge_loss=False,
                m_mask= 0.5
            )
        build_training_functions(**loss_config)

    elif gen_iterations < 0.4 * TOTAL_ITERS:
        if schedule_step == 3:
            return
        schedule_step = 3

        set_batch_size(standard_batch_size)
        loss_config = dict(
                use_PL=True,
                use_mask_hinge_loss=True,
                m_mask= 0.5
            )
        build_training_functions(**loss_config)

    elif gen_iterations < 0.5 * TOTAL_ITERS:
        if schedule_step == 4:
            return
        schedule_step = 4

        set_batch_size(standard_batch_size)
        loss_config = dict(
                use_PL=True,
                use_mask_hinge_loss=True,
                m_mask= 0.25
            )        
        build_training_functions(**loss_config) 

    elif gen_iterations < 0.65 * TOTAL_ITERS:
        if schedule_step == 5:
            return
        schedule_step = 5

        set_batch_size(max_batch_size)
        loss_config = dict(
                use_PL=True,
                use_mask_hinge_loss=True,
                m_mask= 0.33
        )
        build_training_functions(**loss_config)

    else:
        if schedule_step == 6:
            return
        schedule_step = 6

        set_batch_size(max_batch_size)
        loss_config = dict(
                use_PL=True,
                use_mask_hinge_loss=True,
                m_mask= 0.33
        )
        build_training_functions(**loss_config)



    print("Current schedule step: {}".format(schedule_step))

from scipy import ndimage



def get_motion_blur_kernel(sz=7):
    rot_angle = np.random.uniform(-180,180)
    kernel = np.zeros((sz,sz))
    kernel[int((sz-1)//2), :] = np.ones(sz)
    kernel = ndimage.interpolation.rotate(kernel, rot_angle, reshape=False)
    kernel = np.clip(kernel, 0, 1)
    normalize_factor = 1 / np.sum(kernel)
    kernel = kernel * normalize_factor
    return kernel

def motion_blur(images, sz=7):
  # images is a list [image2, image2, ...]
    blur_sz = np.random.choice([5, 7, 9, 11])
    kernel_motion_blur = get_motion_blur_kernel(blur_sz)
    for i, image in enumerate(images):
        images[i] = cv2.filter2D(image, -1, kernel_motion_blur).astype(np.float64)
    return images


def load_data(file_pattern):
    return glob.glob(file_pattern)

def random_channel_shift(x, intensity=10):
    x = cv2.cvtColor(x, cv2.COLOR_BGR2HSV).astype('float32')
    x += np.array([np.random.uniform(-intensity, intensity), 
                   np.random.uniform(-3*intensity, 3*intensity), 
                   np.random.uniform(-3*intensity, 3*intensity)])
    x = np.clip(x, 0, 255).astype('uint8')
    x = cv2.cvtColor(x, cv2.COLOR_HSV2BGR)
    return x
  
def random_cutout(x, intensity=0.2):
    h, w, c = x.shape
    cutout_range = np.min([h//2, w//2])
    cutout_x0 = np.random.randint(h)
    cutout_y0 = np.random.randint(w)
    cutout_x1 = np.min([cutout_x0+np.random.randint(cutout_range)+10, h])
    cutout_y1 = np.min([cutout_y0+np.random.randint(cutout_range)+10, w])
    cutout_h = cutout_x1 - cutout_x0
    cutout_w = cutout_y1 - cutout_y0
    x[cutout_x0:cutout_x1, cutout_y0:cutout_y1, :] = np.random.normal(0., intensity, size=(cutout_h, cutout_w, c))
    return x

def random_warp_rev(image):
    assert image.shape == (256,256,3)
    rand_coverage = np.random.randint(16) + 80
    range_ = np.linspace(128-rand_coverage, 128+rand_coverage, 5)
    mapx = np.broadcast_to(range_, (5,5))
    mapy = mapx.T
    mapx = mapx + np.random.normal( size=(5,5), scale=6)
    mapy = mapy + np.random.normal( size=(5,5), scale=6)
    interp_mapx = cv2.resize( mapx, (80,80) )[8:72,8:72].astype('float32')
    interp_mapy = cv2.resize( mapy, (80,80) )[8:72,8:72].astype('float32')
    warped_image = cv2.remap(image, interp_mapx, interp_mapy, cv2.INTER_LINEAR)
    src_points = np.stack( [ mapx.ravel(), mapy.ravel() ], axis=-1)
    dst_points = np.mgrid[0:65:16,0:65:16].T.reshape(-1,2)
    mat = umeyama(src_points, dst_points, True)[0:2]
    target_image = cv2.warpAffine(image, mat, (64,64))
    return warped_image, target_image

random_transform_args = {
    'rotation_range': 20,
    'zoom_range': 0.15,
    'shift_range': 0.05,
    'random_flip': 0.5,
    }
def read_image(fn, random_transform_args=random_transform_args):
    image = cv2.imread(fn)
    image = cv2.resize(image, (256,256)) / 255 * 2 - 1
    image = random_transform( image, **random_transform_args)
    warped_img, target_img = random_warp_rev(image)    
    
    #if np.random.uniform() < 0.1: # random downscaling
    #    rand_size = np.random.randint(48) + 16
    #    warped_img = cv2.resize(warped_img, (rand_size,rand_size))
    #    warped_img = cv2.resize(warped_img, (64,64))
    #elif np.random.uniform() < 0.3:
    #    warped_img = random_cutout(warped_img)
    
    # Motion blur data augmentation:
    # we want the model to learn to preserve motion blurs of input images
    #if np.random.uniform() < 0.4: 
    #    warped_img, target_img = motion_blur([warped_img, target_img])
    
    return warped_img, target_img


@background(32)
def minibatch(data, size):
    global epoch
    length = len(data)
    i = 0
    shuffle(data)
    epoch1 = epoch
    while True:
        try:
            if i+size > length:
                shuffle(data)
                i = 0
                epoch1+=1        

            rtn = np.float32([read_image(data[j]) for j in range(i,i+size)])
            i+=size
        except Exception as e:
            for j in range(i,i+size):
                print(data[j])
                read_image(data[j])
            raise e
        yield epoch1, rtn[:,0,:,:,:], rtn[:,1,:,:,:] 

def showG(test_A, test_B, path_A, path_B, filename):
    global batchSize
    if base_model == "VAE-GAN" or base_model == "GAN":
        figure_A = np.stack([
            test_A,
            np.squeeze(np.array([path_A([test_A[i:i+1], test_A[i:i+1]]) for i in range(test_A.shape[0])])),
            np.squeeze(np.array([path_B([test_A[i:i+1], test_A[i:i+1]]) for i in range(test_A.shape[0])])),
            ], axis=1 )
        figure_B = np.stack([
            test_B,
            np.squeeze(np.array([path_B([test_B[i:i+1], test_B[i:i+1]]) for i in range(test_B.shape[0])])),
            np.squeeze(np.array([path_A([test_B[i:i+1], test_B[i:i+1]]) for i in range(test_B.shape[0])])),
            ], axis=1 )
    elif base_model == "XGAN":
        figure_A = np.stack([
            test_A,
            np.squeeze(np.array([path_A([test_A[i:i+1]]) for i in range(test_A.shape[0])])),
            np.squeeze(np.array([path_B([test_A[i:i+1]]) for i in range(test_A.shape[0])])),
            ], axis=1 )
        figure_B = np.stack([
            test_B,
            np.squeeze(np.array([path_B([test_B[i:i+1]]) for i in range(test_B.shape[0])])),
            np.squeeze(np.array([path_A([test_B[i:i+1]]) for i in range(test_B.shape[0])])),
            ], axis=1 )

    figure = np.concatenate([figure_A, figure_B], axis=0 )
    figure = figure.reshape((4,batchSize // 2) + figure.shape[1:])
    figure = stack_images(figure)
    figure = np.clip((figure + 1) * 255 / 2, 0, 255).astype('uint8')

    cv2.imwrite(filename, figure)
    
    figure = cv2.cvtColor(figure, cv2.COLOR_BGR2RGB)
    
def showG_mask(test_A, test_B, path_A, path_B, filename):
    global batchSize
    if base_model == "VAE-GAN" or base_model == "GAN":
        figure_A = np.stack([
            test_A,
            (np.squeeze(np.array([path_A([test_A[i:i+1], test_A[i:i+1]]) for i in range(test_A.shape[0])])))*2-1,
            (np.squeeze(np.array([path_B([test_A[i:i+1], test_A[i:i+1]]) for i in range(test_A.shape[0])])))*2-1,
            ], axis=1 )
        figure_B = np.stack([
            test_B,
            (np.squeeze(np.array([path_B([test_B[i:i+1], test_B[i:i+1]]) for i in range(test_B.shape[0])])))*2-1,
            (np.squeeze(np.array([path_A([test_B[i:i+1], test_B[i:i+1]]) for i in range(test_B.shape[0])])))*2-1,
            ], axis=1 )
    elif base_model == "XGAN":
        figure_A = np.stack([
            test_A,
            (np.squeeze(np.array([path_A([test_A[i:i+1]]) for i in range(test_A.shape[0])])))*2-1,
            (np.squeeze(np.array([path_B([test_A[i:i+1]]) for i in range(test_A.shape[0])])))*2-1,
            ], axis=1 )
        figure_B = np.stack([
            test_B,
            (np.squeeze(np.array([path_B([test_B[i:i+1]]) for i in range(test_B.shape[0])])))*2-1,
            (np.squeeze(np.array([path_A([test_B[i:i+1]]) for i in range(test_B.shape[0])])))*2-1,
            ], axis=1 )

    figure = np.concatenate([figure_A, figure_B], axis=0 )
    figure = figure.reshape((4,batchSize // 2) + figure.shape[1:])
    figure = stack_images(figure)
    figure = np.clip((figure + 1) * 255 / 2, 0, 255).astype('uint8')
    
    cv2.imwrite(filename, figure)

    figure = cv2.cvtColor(figure, cv2.COLOR_BGR2RGB)



train_A = load_data(img_dirA)
train_B = load_data(img_dirB)

assert len(train_A), "No image found in " + str(img_dirA)
assert len(train_B), "No image found in " + str(img_dirB)

print ("Number of images in folder of face A: " + str(len(train_A)))
print ("Number of images in folder of face B: " + str(len(train_B)))


progress = load_progress(model_dir)
if 'iterations' in progress and len(progress['iterations']) > 0 :
    gen_iterations = progress['iterations'][-1] + 1
    print("Restarting from iteration {}".format(gen_iterations))
else:
    gen_iterations = 0

if 'epoch' in progress and len(progress['epoch']) > 0:
    epoch = progress['epoch'][-1]
else:
    epoch = 0

errGA_sum = errGB_sum = errDA_sum = errDB_sum = errDA2_sum = errDB2_sum = errDA_code_sum = errDB_code_sum = 0

display_iters = 100
train_batchA = minibatch(train_A, batchSize)
train_batchB = minibatch(train_B, batchSize)

TOTAL_ITERS = 40000


schedule_training_functions(gen_iterations, display_iters, TOTAL_ITERS)

while gen_iterations < TOTAL_ITERS: 

    t0 = time.time()
    epoch, warped_A, target_A = next(train_batchA) 
    epoch, warped_B, target_B = next(train_batchB) 

    # Train dicriminators for one batch
    if gen_iterations % 1 == 0:
        if base_model == "VAE-GAN" or base_model == "GAN":
            errDA = netDA_train([warped_A, warped_A, target_A])
            errDB = netDB_train([warped_B, warped_B, target_B])
            errDA2 = netDA2_train([warped_A, warped_A, target_A])
            errDB2 = netDB2_train([warped_B, warped_B, target_B])
        elif base_model == "XGAN":
            errDA = netDA_train([warped_A, target_A])
            errDB = netDB_train([warped_B, target_B])
            errDA2 = netDA2_train([warped_A, target_A])
            errDB2 = netDB2_train([warped_B, target_B])
            errDA_code = netDA_code_train([warped_A, target_A])
            errDB_code = netDB_code_train([warped_B, target_B])
    errDA_sum += errDA[0]
    errDB_sum += errDB[0]
    errDA2_sum += errDA2[0]
    errDB2_sum += errDB2[0]
    if base_model == "XGAN":
        errDA_code_sum += errDA_code[0]
        errDB_code_sum += errDB_code[0]


    
    # Train generators for one batch
    if base_model == "VAE-GAN" or base_model == "GAN":
        errGA = netGA_train([warped_A, warped_A, target_A])
        errGB = netGB_train([warped_B, warped_B, target_B])
    elif base_model == "XGAN":
        errGA = netGA_train([warped_A, target_A])
        errGB = netGB_train([warped_B, target_B])
    errGA_sum += errGA[0]
    errGB_sum += errGB[0]
    gen_iterations += 1
    

    if gen_iterations % display_iters != 0:
        print('[epoch %d][iter %d] Loss_DA: %f Loss_DB: %f Loss_DA2: %f Loss_DB2: %f Loss_GA: %f Loss_GB: %f time: %dms'
        % (epoch, gen_iterations, errDA[0], errDB[0], errDA2[0], errDB2[0],
           errGA[0], errGB[0], int(1000*(time.time()-t0))))    
        
        progress['epoch'].append(int(epoch))
        progress['iterations'].append(int(gen_iterations))
        progress['errDA'].append(float(errDA[0]))
        progress['errDB'].append(float(errDB[0]))
        progress['errDA2'].append(float(errDA2[0]))
        progress['errDB2'].append(float(errDB2[0]))
        progress['errGA'].append(float(errGA[0]))
        progress['errGB'].append(float(errGB[0]))
        progress['base_model'] = base_model
    
        

    else:
        
        print('[epoch %d][iter %d] Loss_DA: %f Loss_DB: %f Loss_DA2: %f Loss_DB2: %f Loss_GA: %f Loss_GB: %f time: %dms'
        % (epoch, gen_iterations, errDA_sum/display_iters, errDB_sum/display_iters, errDA2_sum/display_iters, errDB2_sum/display_iters,
           errGA_sum/display_iters, errGB_sum/display_iters, int(1000*(time.time()-t0))))
        if base_model == "XGAN":
            print('Loss_DA_code: %f Loss_DB_code: %f' % (errDA_code_sum/display_iters, errDB_code_sum/display_iters)) 
        
        # Show previews
        print("Creating previews")
        _, wA, tA = train_batchA.next()
        _, wB, tB = train_batchB.next()

        fn = str(preview_dir / 'preview_{:06d}_masked.png'.format(gen_iterations))
        showG(tA, tB, path_A, path_B, fn)

        fn = str(preview_dir / 'preview_{:06d}_raw.png'.format(gen_iterations))
        showG(wA, wB, path_bgr_A, path_bgr_B, fn)   
        
        fn = str(preview_dir / 'preview_{:06d}_mask.png'.format(gen_iterations))
        showG_mask(tA, tB, path_mask_A, path_mask_B, fn)  

        errGA_sum = errGB_sum = errDA_sum = errDB_sum = 0
        errDA2_sum = errDB2_sum = 0
        if base_model == "XGAN":
            errDA_code_sum = errDB_code_sum = 0

    if gen_iterations % (5 * display_iters) == 0:
    
        save_models(model_dir)
        save_progress(model_dir, progress)
        plot_progress(model_dir, progress)


    schedule_training_functions(gen_iterations, display_iters, TOTAL_ITERS)

# # 11. Video Conversion

from moviepy.editor import VideoFileClip



path_func = path_abgr_A


# ## MTCNN 

import mtcnn_detect_face


def create_mtcnn(sess, model_path):
    if not model_path:
        model_path,_ = os.path.split(os.path.realpath(__file__))

    with tf.variable_scope('pnet2'):
        data = tf.placeholder(tf.float32, (None,None,None,3), 'input')
        pnet = mtcnn_detect_face.PNet({'data':data})
        pnet.load(os.path.join(model_path, 'det1.npy'), sess)
    with tf.variable_scope('rnet2'):
        data = tf.placeholder(tf.float32, (None,24,24,3), 'input')
        rnet = mtcnn_detect_face.RNet({'data':data})
        rnet.load(os.path.join(model_path, 'det2.npy'), sess)
    with tf.variable_scope('onet2'):
        data = tf.placeholder(tf.float32, (None,48,48,3), 'input')
        onet = mtcnn_detect_face.ONet({'data':data})
        onet.load(os.path.join(model_path, 'det3.npy'), sess)
    return pnet, rnet, onet


WEIGHTS_PATH = "./mtcnn_weights/"

sess = K.get_session()
with sess.as_default():
    global pnet, rnet, onet 
    pnet2, rnet2, onet2 = create_mtcnn(sess, WEIGHTS_PATH)


global pnet, rnet, onet
pnet_fun = K.function([pnet2.layers['data']],[pnet2.layers['conv4-2'], pnet2.layers['prob1']])
rnet_fun = K.function([rnet2.layers['data']],[rnet2.layers['conv5-2'], rnet2.layers['prob1']])
onet_fun = K.function([onet2.layers['data']], [onet2.layers['conv6-2'], onet2.layers['conv6-3'], onet2.layers['prob1']])

with tf.variable_scope('pnet2', reuse=True):
    data = tf.placeholder(tf.float32, (None,None,None,3), 'input')
    pnet2 = mtcnn_detect_face.PNet({'data':data})
    pnet2.load(os.path.join("./mtcnn_weights/", 'det1.npy'), sess)
with tf.variable_scope('rnet2', reuse=True):
    data = tf.placeholder(tf.float32, (None,24,24,3), 'input')
    rnet2 = mtcnn_detect_face.RNet({'data':data})
    rnet2.load(os.path.join("./mtcnn_weights/", 'det2.npy'), sess)
with tf.variable_scope('onet2', reuse=True):
    data = tf.placeholder(tf.float32, (None,48,48,3), 'input')
    onet2 = mtcnn_detect_face.ONet({'data':data})
    onet2.load(os.path.join("./mtcnn_weights/", 'det3.npy'), sess)



pnet = K.function([pnet2.layers['data']],[pnet2.layers['conv4-2'], pnet2.layers['prob1']])
rnet = K.function([rnet2.layers['data']],[rnet2.layers['conv5-2'], rnet2.layers['prob1']])
onet = K.function([onet2.layers['data']], [onet2.layers['conv6-2'], onet2.layers['conv6-3'], onet2.layers['prob1']])


# ## FCN8s for face segmentation
from FCN8s_keras import FCN

fcn = FCN()

print("Load FCN weights")
fcn.load_weights("Keras_FCN8s_face_seg_YuvalNirkin.h5")

def fcn_mask_seg(x):
    def vgg_preprocess(im):
        im = cv2.resize(im, (500, 500))
        in_ = np.array(im, dtype=np.float32)
        #in_ = in_[:,:,::-1]
        in_ -= np.array((104.00698793,116.66876762,122.67891434))
        in_ = in_[np.newaxis,:]
        #in_ = in_.transpose((2,0,1))
        return in_
    
    inp_fcn = vgg_preprocess(x)
    out_mask_fcn = fcn.predict([inp_fcn])
    out_mask_fcn = cv2.resize(np.squeeze(out_mask_fcn), (x.shape[1],x.shape[0]))
    out_mask_fcn = np.clip(out_mask_fcn.argmax(axis=2), 0, 1).astype(np.float64)
    return out_mask_fcn


# ## Video conversion functions
use_smoothed_mask = True
use_smoothed_bbox = True

def is_overlap(box1, box2):
    overlap_x0 = np.max([box1[0], box2[0]]).astype(np.float32)
    overlap_y1 = np.min([box1[1], box2[1]]).astype(np.float32)
    overlap_x1 = np.min([box1[2], box2[2]]).astype(np.float32)
    overlap_y0 = np.max([box1[3], box2[3]]).astype(np.float32)
    area_iou = (overlap_x1-overlap_x0) * (overlap_y1-overlap_y0)
    area_box1 = (box1[2]-box1[0]) * (box1[1]-box1[3])
    area_box2 = (box2[2]-box2[0]) * (box2[1]-box2[3])    
    return (area_iou / area_box1) >= 0.2
    
def remove_overlaps(faces):    
    main_face = get_most_conf_face(faces)
    main_face_bbox = main_face[0]
    result_faces = []
    result_faces.append(main_face_bbox)
    for (x0, y1, x1, y0, conf_score) in faces:
        if not is_overlap(main_face_bbox, (x0, y1, x1, y0)):
            result_faces.append((x0, y1, x1, y0, conf_score))
    return result_faces

def get_most_conf_face(faces):
    # Return the bbox w/ the highest confidence score
    best_conf_score = 0
    conf_face = None
    for (x0, y1, x1, y0, conf_score) in faces: 
        if conf_score >= best_conf_score:
            best_conf_score = conf_score
            conf_face = [(x0, y1, x1, y0, conf_score)]
    return conf_face

def kalmanfilter_init(noise_coef):
    kf = cv2.KalmanFilter(4,2)
    kf.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
    kf.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)
    kf.processNoiseCov = noise_coef * np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]], np.float32)
    return kf

def is_higher_than_480p(x):
    return (x.shape[0] * x.shape[1]) >= (858*480)

def is_higher_than_720p(x):
    return (x.shape[0] * x.shape[1]) >= (1280*720)

def is_higher_than_1080p(x):
    return (x.shape[0] * x.shape[1]) >= (1920*1080)

def calibrate_coord(faces, video_scaling_factor):
    for i, (x0, y1, x1, y0, _) in enumerate(faces):
        faces[i] = (x0*video_scaling_factor, y1*video_scaling_factor, 
                    x1*video_scaling_factor, y0*video_scaling_factor, _)
    return faces

def process_mtcnn_bbox(bboxes, im_shape):
    # outuut bbox coordinate of MTCNN is (y0, x0, y1, x1)
    # Process the bbox coord. to a square bbox with ordering (x0, y1, x1, y0)
    for i, bbox in enumerate(bboxes):
        y0, x0, y1, x1 = bboxes[i,0:4]
        w = int(y1 - y0)
        h = int(x1 - x0)
        length = (w + h) / 2
        center = (int((x1+x0)/2),int((y1+y0)/2))
        new_x0 = np.max([0, (center[0]-length//2)])#.astype(np.int32)
        new_x1 = np.min([im_shape[0], (center[0]+length//2)])#.astype(np.int32)
        new_y0 = np.max([0, (center[1]-length//2)])#.astype(np.int32)
        new_y1 = np.min([im_shape[1], (center[1]+length//2)])#.astype(np.int32)
        bboxes[i,0:4] = new_x0, new_y1, new_x1, new_y0
    return bboxes

def get_downscale_factor(image):
    if is_higher_than_1080p(image):
        return  4 + video_scaling_offset
    elif is_higher_than_720p(image):
        return 3 + video_scaling_offset
    elif is_higher_than_480p(image):
        return 2 + video_scaling_offset
    else:
        return 1

def get_faces_bbox(image):  
    global pnet, rnet, onet 
    global detec_threshold
    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, detec_threshold ]  # three steps's threshold
    factor = 0.709 # scale factor
    if manually_downscale:
        video_scaling_factor = manual_downscale_factor
        resized_image = cv2.resize(image, 
                                   (image.shape[1]//video_scaling_factor, 
                                    image.shape[0]//video_scaling_factor))
        faces, pnts = mtcnn_detect_face.detect_face(resized_image, minsize, pnet, rnet, onet, threshold, factor)
        faces = process_mtcnn_bbox(faces, resized_image.shape)
        faces = calibrate_coord(faces, video_scaling_factor)
    else:
        video_scaling_factor = get_downscale_factor(image)
        resized_image = cv2.resize(image, 
                                   (image.shape[1]//video_scaling_factor, 
                                    image.shape[0]//video_scaling_factor))
        faces, pnts = mtcnn_detect_face.detect_face(resized_image, minsize, pnet, rnet, onet, threshold, factor)
        faces = process_mtcnn_bbox(faces, resized_image.shape)
        faces = calibrate_coord(faces, video_scaling_factor)
    return faces

def get_smoothed_coord(x0, x1, y0, y1, shape, ratio=0.65):
    global prev_x0, prev_x1, prev_y0, prev_y1
    if not use_kalman_filter:
        x0 = int(ratio * prev_x0 + (1-ratio) * x0)
        x1 = int(ratio * prev_x1 + (1-ratio) * x1)
        y1 = int(ratio * prev_y1 + (1-ratio) * y1)
        y0 = int(ratio * prev_y0 + (1-ratio) * y0)
    else:
        x0y0 = np.array([x0, y0]).astype(np.float32)
        x1y1 = np.array([x1, y1]).astype(np.float32)
        kf0.correct(x0y0)
        pred_x0y0 = kf0.predict()
        kf1.correct(x1y1)
        pred_x1y1 = kf1.predict()
        x0 = np.max([0, pred_x0y0[0][0]]).astype(np.int)
        x1 = np.min([shape[0], pred_x1y1[0][0]]).astype(np.int)
        y0 = np.max([0, pred_x0y0[1][0]]).astype(np.int)
        y1 = np.min([shape[1], pred_x1y1[1][0]]).astype(np.int)
        if x0 == x1 or y0 == y1:
            x0, y0, x1, y1 = prev_x0, prev_y0, prev_x1, prev_y1
    return x0, x1, y0, y1     
    
def set_global_coord(x0, x1, y0, y1):
    global prev_x0, prev_x1, prev_y0, prev_y1
    prev_x0 = x0
    prev_x1 = x1
    prev_y1 = y1
    prev_y0 = y0
    
def merge_mask(mask1, mask2,mode="avg"):
    mask2 = cv2.resize(mask2, (mask1.shape[1],mask1.shape[0]))
    if mode == "avg":
        return (mask1 + mask2).astype(np.float32) / 2
    elif mode == "max":
        return np.maximum(mask1, mask2)
    elif mode == "adaptive":
        sum1 = np.sum(mask1)/3
        sum2 = np.sum(mask2)
        ratio = np.clip(sum1/(sum1+sum2), 0.2, 0.8)
        return ratio*mask1 + (1-ratio)*mask2

def get_tta_masked_result(result_bgr, result_a, roi_image):
    result_bgr = cv2.resize(result_bgr, (64,64)).astype(np.float64)
    result_a = cv2.resize(result_a, (64,64)).astype(np.float64)
    roi_image_resized = cv2.resize(roi_image, (64,64)).astype(np.float64)
    result_a = cv2.GaussianBlur(result_a, (7,7), 6)[:,:,np.newaxis]
    result = (result_a/255 * result_bgr.astype(np.float64) + (1-result_a/255) * roi_image_resized).astype(np.uint8)
    result = cv2.resize(result, (roi_image.shape[1],roi_image.shape[0]))
    if use_color_correction:
        result = color_hist_match(result, roi_image).astype('uint8')
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB).astype('uint8')
    return result    

def generate_face(ae_input, path_abgr, roi_size, roi_image):
    if base_model == "XGAN":
        result = np.squeeze(np.array([path_abgr([[ae_input]])]))
    elif base_model == "VAE-GAN" or base_model =="GAN":
        result = np.squeeze(np.array([path_abgr([[ae_input], [ae_input]])]))
    result_a = result[:,:,0] * 255
    result_bgr = np.clip( (result[:,:,1:] + 1) * 255 / 2, 0, 255 )
    result_a_clear = np.copy(result_a)
    result_a = cv2.GaussianBlur(result_a ,(7,7),6)
    if use_landmark_match and False:
        resized_roi = cv2.resize(roi_image, (64,64))
        result_bgr, result_a = landmarks_match_mtcnn(resized_roi, result_bgr, result_a)
    #if use_color_correction:
    #    result_bgr = color_hist_match(result_bgr, roi_image)
    
    if use_FCN_mask:
        fcn_mask = fcn_mask_seg(roi_image)
        if merge_mask_mode == "avg":
            fcn_mask = merge_mask(fcn_mask, result[:,:,0], mode="avg")
        elif merge_mask_mode == "max":
            fcn_mask = merge_mask(fcn_mask, result[:,:,0], mode="max")
        elif merge_mask_mode == "adaptive":
            fcn_mask = merge_mask(fcn_mask, result[:,:,0], mode="adaptive")
        result_a = fcn_mask[:,:,np.newaxis] * 255
        result_a_clear = np.copy(fcn_mask * 255)
        fcn_mask_sz64 = cv2.resize(fcn_mask, (64,64))
        fcn_mask_sz64 = cv2.GaussianBlur(fcn_mask_sz64, (7,7), 6)[:,:,np.newaxis]
        result = (fcn_mask_sz64 * result_bgr.astype(np.float64) + (1-fcn_mask_sz64) * ((ae_input+1)*255/2)).astype(np.uint8)
    else:
        result_a = np.expand_dims(result_a, axis=2)
        result = (result_a/255 * result_bgr + (1 - result_a/255) * ((ae_input + 1) * 255 / 2)).astype('uint8')
        result = result_bgr.astype('uint8')
    if use_color_correction:
        result = color_hist_match(result, roi_image).astype('uint8')
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    result = cv2.resize(result, (roi_size[1],roi_size[0]))
    result_a_clear = np.expand_dims(cv2.resize(result_a_clear, (roi_size[1],roi_size[0])), axis=2)
    return result, result_a_clear, result_bgr

def get_init_mask_map(image):
    return np.zeros_like(image)

def get_init_comb_img(input_img):
    comb_img = np.zeros([input_img.shape[0], input_img.shape[1]*2,input_img.shape[2]])
    comb_img[:, :input_img.shape[1], :] = input_img
    comb_img[:, input_img.shape[1]:, :] = input_img
    return comb_img    

def get_init_triple_img(input_img, no_face=False):
    if no_face:
        triple_img = np.zeros([input_img.shape[0], input_img.shape[1]*3,input_img.shape[2]])
        triple_img[:, :input_img.shape[1], :] = input_img
        triple_img[:, input_img.shape[1]:input_img.shape[1]*2, :] = input_img      
        triple_img[:, input_img.shape[1]*2:, :] = (input_img * .15).astype('uint8')  
        return triple_img
    else:
        triple_img = np.zeros([input_img.shape[0], input_img.shape[1]*3,input_img.shape[2]])
        return triple_img

def get_mask(roi_image, h, w):
    mask = np.zeros_like(roi_image)
    mask[h//15:-h//15,w//15:-w//15,:] = 255
    mask = cv2.GaussianBlur(mask,(15,15),10)
    return mask

def hist_match(source, template):
    # Code borrow from:
    # https://stackoverflow.com/questions/32655686/histogram-matching-of-two-images-in-python-2-x
    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)

def color_hist_match(src_im, tar_im):
    #src_im = cv2.cvtColor(src_im, cv2.COLOR_BGR2HSV)
    #tar_im = cv2.cvtColor(tar_im, cv2.COLOR_BGR2HSV)
    matched_0 = hist_match(src_im[:,:,0], tar_im[:,:,0])
    matched_1 = hist_match(src_im[:,:,1], tar_im[:,:,1])
    matched_2 = hist_match(src_im[:,:,2], tar_im[:,:,2])
    matched = np.stack((matched_1, matched_1, matched_2), axis=2).astype("uint8")
    #matched = cv2.cvtColor(matched, cv2.COLOR_HSV2BGR)
    return matched

def landmarks_match_mtcnn(source, target, alpha):
    global prev_pnts1, prev_pnts2
    ratio = 0.2
    """
    TODO: Reuse the landmarks of source image. Conceivable bug: coordinate mismatch.
    """
    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.93 ]  # three steps's threshold
    factor = 0.709 # scale factor
    _, pnts1 = mtcnn_detect_face.detect_face(source, minsize, pnet, rnet, onet, threshold, factor) # redundant detection
    _, pnts2 = mtcnn_detect_face.detect_face(target, minsize, pnet, rnet, onet, threshold, factor)  
    
    if len(prev_pnts1) == 0 and len(prev_pnts2) == 0:
        if pnts1.shape[0] == 10 and pnts2.shape[0] == 10:
            prev_pnts1, prev_pnts2 = pnts1, pnts2        
    try:
        landmarks_XY1 = []
        landmarks_XY2 = []
        for i in range(5):
            landmarks_XY1.extend([((1-ratio)*pnts1[i+5][0] + ratio*prev_pnts1[i+5][0], 
                                   (1-ratio)*pnts1[i][0] + ratio*prev_pnts1[i][0])])
            landmarks_XY2.extend([((1-ratio)*pnts2[i+5][0] + ratio*prev_pnts2[i+5][0], 
                                   (1-ratio)*pnts2[i][0] + ratio*prev_pnts2[i][0])])
        M = umeyama(np.array(landmarks_XY1), np.array(landmarks_XY2), True)[0:2]
        result = cv2.warpAffine(source, M, (64, 64), borderMode=cv2.BORDER_REPLICATE)  
        mask = np.stack([alpha, alpha, alpha], axis=2)
        assert len(mask.shape) == 3, "len(mask.shape) is " + str(len(mask.shape))
        mask = cv2.warpAffine(mask, M, (64, 64), borderMode=cv2.BORDER_REPLICATE) 
        prev_landmarks_XY1, prev_landmarks_XY2 = landmarks_XY1, landmarks_XY2
        return result, mask[:,:,0].astype(np.float32)
    except:
        return source, alpha

def process_video(input_img): 
    try:
        global prev_x0, prev_x1, prev_y0, prev_y1
        global frames      
        global pnet, rnet, onet
        """The following if statement is meant to solve a bug that has an unknow cause."""
        if frames <= 2:
            print("Slow bug")
            with tf.variable_scope('pnet2', reuse=True):
                pnet2 = None
                data = tf.placeholder(tf.float32, (None,None,None,3), 'input')
                pnet2 = mtcnn_detect_face.PNet({'data':data})
                pnet2.load(os.path.join("./mtcnn_weights/", 'det1.npy'), sess)
            with tf.variable_scope('rnet2', reuse=True):
                rnet2 = None
                data = tf.placeholder(tf.float32, (None,24,24,3), 'input')
                rnet2 = mtcnn_detect_face.RNet({'data':data})
                rnet2.load(os.path.join("./mtcnn_weights/", 'det2.npy'), sess)
            with tf.variable_scope('onet2', reuse=True):
                onet2 = None
                data = tf.placeholder(tf.float32, (None,48,48,3), 'input')
                onet2 = mtcnn_detect_face.ONet({'data':data})
                onet2.load(os.path.join("./mtcnn_weights/", 'det3.npy'), sess)
            pnet = K.function([pnet2.layers['data']],
                              [pnet2.layers['conv4-2'], 
                               pnet2.layers['prob1']])
            rnet = K.function([rnet2.layers['data']],
                              [rnet2.layers['conv5-2'], 
                               rnet2.layers['prob1']])
            onet = K.function([onet2.layers['data']], 
                              [onet2.layers['conv6-2'], 
                               onet2.layers['conv6-3'], 
                               onet2.layers['prob1']])
        """Get face bboxes"""
        image = input_img
        faces = get_faces_bbox(image) # faces: face bbox coord
        
        """Init."""
        if len(faces) == 0:
            comb_img = get_init_comb_img(input_img)
            triple_img = get_init_triple_img(input_img, no_face=True)
        else:
            _ = remove_overlaps(faces) # Has non-max suppression already been implemented in MTCNN?        
        mask_map = get_init_mask_map(image)
        comb_img = get_init_comb_img(input_img)
        best_conf_score = 0
        
        """Process detected faces"""
        for (x0, y1, x1, y0, conf_score) in faces: 
            """Apply moving average bounding box"""        
            if use_smoothed_bbox:
                if frames != 0 and conf_score >= best_conf_score:
                    x0, x1, y0, y1 = get_smoothed_coord(x0, x1, y0, y1, 
                                                        image.shape, 
                                                        ratio=0.65 if use_kalman_filter else bbox_moving_avg_coef)
                    set_global_coord(x0, x1, y0, y1)
                    best_conf_score = conf_score
                    frames += 1
                elif conf_score <= best_conf_score:
                    frames += 1
                else:
                    if conf_score >= best_conf_score:
                        set_global_coord(x0, x1, y0, y1)
                        best_conf_score = conf_score
                    if use_kalman_filter:
                        for i in range(200):
                            kf0.predict()
                            kf1.predict()
                    frames += 1
            
            """ROI params"""
            h = int(x1 - x0)
            w = int(y1 - y0)
            roi_coef_h = 25
            roi_coef_w = 25
            roi_x0, roi_x1, roi_y0, roi_y1 = int(x0+h//roi_coef_h), int(x1-h//roi_coef_h), int(y0+w//roi_coef_w), int(y1-w//roi_coef_w)            
            cv2_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            roi_image = cv2_img[roi_x0:roi_x1,roi_y0:roi_y1,:]
            roi_size = roi_image.shape  
            
            """Feed face image into generator"""
            ae_input = cv2.resize(roi_image, (64,64))/255. * 2 - 1  
            result, result_a, result_bgr = generate_face(ae_input, path_func, roi_size, roi_image)
            
            """Apply test time augmentation"""
            if use_TTA:
                result_flip, result_a_flip, result_bgr_flip = generate_face(ae_input[:,::-1,:], 
                                                                            path_func, roi_size, 
                                                                            roi_image[:,::-1,:])
                result = (result.astype(np.float32) + result_flip[:,::-1,:].astype(np.float32))/2
                result_a = (result_a.astype(np.float32) + result_a_flip[:,::-1,:].astype(np.float32))/2
                result_bgr = (result_bgr.astype(np.float32) + result_bgr_flip[:,::-1,:].astype(np.float32))/2          
                result = result.astype('uint8')
                result_a = result_a.astype('uint8')
                result = get_tta_masked_result(result_bgr, result_a, roi_image)
                
            """Post processing"""
            if conf_score >= best_conf_score:
                mask_map[roi_x0:roi_x1,roi_y0:roi_y1,:] = result_a
                mask_map = np.clip(mask_map + .15 * input_img, 0, 255)     
            else:
                mask_map[roi_x0:roi_x1,roi_y0:roi_y1,:] += result_a
                mask_map = np.clip(mask_map, 0, 255)        
            if use_smoothed_mask:
                mask = get_mask(roi_image, h, w)
                roi_rgb = cv2.cvtColor(roi_image, cv2.COLOR_BGR2RGB)
                smoothed_result = mask/255 * result + (1-mask/255) * roi_rgb
                comb_img[roi_x0:roi_x1, input_img.shape[1]+roi_y0:input_img.shape[1]+roi_y1,:] = smoothed_result
            else:
                comb_img[roi_x0:roi_x1, input_img.shape[1]+roi_y0:input_img.shape[1]+roi_y1,:] = result
                
            triple_img = get_init_triple_img(input_img)
            triple_img[:, :input_img.shape[1]*2, :] = comb_img
            triple_img[:, input_img.shape[1]*2:, :] = mask_map
        
        """Return frame result"""
        global output_type
        if output_type == 1:
            return comb_img[:, input_img.shape[1]:, :]  # return only result image
        elif output_type == 2:
            return comb_img  # return input and result image combined as one
        elif output_type == 3:
            return triple_img #return input,result and mask heatmap image combined as one
    except:
        return input_img

# ## Video conversion config
use_kalman_filter = True
if use_kalman_filter:
    noise_coef = 8e-2 # Increase by 10x if tracking is slow. 
    kf0 = kalmanfilter_init(noise_coef)
    kf1 = kalmanfilter_init(noise_coef)
else:
    bbox_moving_avg_coef = 0.65
    
video_scaling_offset = 0 
manually_downscale = False
manual_downscale_factor = int(2) # should be an positive integer
use_color_correction = False
use_landmark_match = False # Under developement, This is not functioning.
use_TTA = False # test time augmentation
use_FCN_mask = True
merge_mask_mode = "adaptive" # avg, max, adaptive

# ========== Change the following line for different output type==========
# Output type: 
#    1. [ result ] 
#    2. [ source | result ] 
#    3. [ source | result | mask ]
global output_type
output_type = 1

# Detection threshold:  a float point between 0 and 1. Decrease this value if faces are missed.
global detec_threshold
detec_threshold = 0.7


# ## Start video conversion
# Variables for smoothing bounding box
global prev_x0, prev_x1, prev_y0, prev_y1
global frames
global prev_pnts1, prev_pnts2
prev_x0 = prev_x1 = prev_y0 = prev_y1 = 0
frames = 0
prev_pnts1 = prev_pnts2 = np.array([])


output_dir = target_dir / "swapped"
output_dir.mkdir(parents=True, exist_ok=True)
input_dir = target_dir / "to-swap"

for in_file in input_dir.glob("**/*"):
    print(str(in_file))
    if in_file.is_file():
        try:
            out_file = output_dir / in_file.name
            print(str(out_file))
            clip1 = VideoFileClip(str(in_file))
            clip = clip1.fl_image(process_video)#.subclip(1, 10) #NOTE: this function expects color images!!
            clip.write_videofile(str(out_file), audio=True)
        except Exception as e:
            print(e)
            

