from keras.models import Model
from keras.layers import *
from keras.optimizers import Adam
from .nn_blocks import *
from .losses import *

"""
faceswap-GAN v2.2 model
"""

class FaceswapGANModel():
    def __init__(self, **arch_config):
        self.nc_G_inp = 3
        self.nc_D_inp = 6 
        self.IMAGE_SHAPE = (64, 64, 3)
        self.lrD = 2e-4
        self.lrG = 1e-4
        self.use_self_attn = arch_config['use_self_attn']
        self.norm = arch_config['norm']
        self.model_capacity = arch_config['model_capacity']
        
        # define networks
        self.encoder = self.build_encoder(nc_in=self.nc_G_inp, 
                                          input_size=64, 
                                          use_self_attn=self.use_self_attn,
                                          norm=self.norm,
                                          model_capacity=self.model_capacity
                                         )
        self.decoder_A = self.build_decoder(nc_in=512, 
                                            input_size=8, 
                                            use_self_attn=self.use_self_attn,
                                            norm=self.norm,
                                            model_capacity=self.model_capacity
                                           )
        self.decoder_B = self.build_decoder(nc_in=512, 
                                            input_size=8, 
                                            use_self_attn=self.use_self_attn,
                                            norm=self.norm,
                                            model_capacity=self.model_capacity
                                           )
        self.netDA = self.build_discriminator(nc_in=self.nc_D_inp, 
                                              input_size=64,
                                              use_self_attn=self.use_self_attn,
                                              norm=self.norm                                         
                                             )
        self.netDB = self.build_discriminator(nc_in=self.nc_D_inp, 
                                              input_size=64,
                                              use_self_attn=self.use_self_attn,
                                              norm=self.norm                                         
                                             )
        x = Input(shape=self.IMAGE_SHAPE) # dummy input tensor
        self.netGA = Model(x, self.decoder_A(self.encoder(x)))
        self.netGB = Model(x, self.decoder_B(self.encoder(x)))
        
        # define variables
        self.distorted_A, self.fake_A, self.mask_A, \
        self.path_A, self.path_mask_A, self.path_abgr_A, self.path_bgr_A = self.define_variables(netG=self.netGA)
        self.distorted_B, self.fake_B, self.mask_B, \
        self.path_B, self.path_mask_B, self.path_abgr_B, self.path_bgr_B = self.define_variables(netG=self.netGB)
        self.real_A = Input(shape=self.IMAGE_SHAPE)
        self.real_B = Input(shape=self.IMAGE_SHAPE)
        self.mask_eyes_A = Input(shape=self.IMAGE_SHAPE)
        self.mask_eyes_B = Input(shape=self.IMAGE_SHAPE)
    
    @staticmethod
    def build_encoder(nc_in=3, input_size=64, use_self_attn=True, norm='none', model_capacity='standard'):
        coef = 2 if model_capacity == "lite" else 1
        
        inp = Input(shape=(input_size, input_size, nc_in))
        x = Conv2D(64, kernel_size=5, use_bias=False, padding="same")(inp)
        x = conv_block(x, 128)
        x = conv_block(x, 256, True, norm=norm)
        x = self_attn_block(x, 256) if use_self_attn else x
        x = conv_block(x, 512, True, norm=norm) 
        x = self_attn_block(x, 512) if use_self_attn else x
        x = conv_block(x, 1024//coef, True, norm=norm)
        x = Dense(1024//coef)(Flatten()(x))
        x = Dense(4*4*1024//coef)(x)
        x = Reshape((4, 4, 1024//coef))(x)
        out = upscale_ps(x, 512, True, norm=norm)
        return Model(inputs=inp, outputs=out)        
    
    @staticmethod
    def build_decoder(nc_in=512, input_size=8, use_self_attn=True, norm='none', model_capacity='standard'):  
        coef = 2 if model_capacity == "lite" else 1

        inp = Input(shape=(input_size, input_size, nc_in))
        x = inp
        x = upscale_ps(x, 256, True, norm=norm)
        x = upscale_ps(x, 128, True, norm=norm)
        x = self_attn_block(x, 128) if use_self_attn else x
        x = upscale_ps(x, 64//coef, True, norm=norm)
        x = res_block(x, 64//coef, norm=norm)
        x = self_attn_block(x, 64//coef) if use_self_attn else conv_block(x, 64//coef, strides=1)
        alpha = Conv2D(1, kernel_size=5, padding='same', activation="sigmoid")(x)
        bgr = Conv2D(3, kernel_size=5, padding='same', activation="tanh")(x)
        out = concatenate([alpha, bgr])
        return Model(inp, out)
    
    @staticmethod
    def build_discriminator(nc_in, input_size=64, use_self_attn=True, norm='none'):    
        inp = Input(shape=(input_size, input_size, nc_in))
        x = conv_block_d(inp, 64, False)
        x = conv_block_d(x, 128, True, norm=norm)
        x = conv_block_d(x, 256, True, norm=norm)
        x = self_attn_block(x, 256) if use_self_attn else x
        out = Conv2D(1, kernel_size=4, use_bias=False, padding="same")(x)   
        return Model(inputs=[inp], outputs=out)
    
    @staticmethod
    def define_variables(netG):
        distorted_input = netG.inputs[0]
        fake_output = netG.outputs[0]
        alpha = Lambda(lambda x: x[:,:,:, :1])(fake_output)
        bgr = Lambda(lambda x: x[:,:,:, 1:])(fake_output)

        masked_fake_output = alpha * bgr + (1-alpha) * distorted_input 

        fn_generate = K.function([distorted_input], [masked_fake_output])
        fn_mask = K.function([distorted_input], [concatenate([alpha, alpha, alpha])])
        fn_abgr = K.function([distorted_input], [concatenate([alpha, bgr])])
        fn_bgr = K.function([distorted_input], [bgr])
        return distorted_input, fake_output, alpha, fn_generate, fn_mask, fn_abgr, fn_bgr 
    
    def build_train_functions(self, loss_weights=None, **loss_config):
        assert loss_weights is not None, "loss weights are not provided."
        # Adversarial loss
        loss_DA, loss_adv_GA = adversarial_loss(self.netDA, self.real_A, self.fake_A, 
                                                self.distorted_A, **loss_weights)
        loss_DB, loss_adv_GB = adversarial_loss(self.netDB, self.real_B, self.fake_B, 
                                                self.distorted_B, **loss_weights)

        # Reconstruction loss
        loss_recon_GA = reconstruction_loss(self.real_A, self.fake_A, 
                                            self.mask_eyes_A, **loss_weights)
        loss_recon_GB = reconstruction_loss(self.real_B, self.fake_B, 
                                            self.mask_eyes_B, **loss_weights)

        # Edge loss
        loss_edge_GA = edge_loss(self.real_A, self.fake_A, self.mask_eyes_A, **loss_weights)
        loss_edge_GB = edge_loss(self.real_B, self.fake_B, self.mask_eyes_B, **loss_weights)

        if loss_config['use_PL']:
            loss_pl_GA = perceptual_loss(self.real_A, self.fake_A, self.distorted_A, 
                                         self.mask_eyes_A, self.vggface_feats, **loss_weights)
            loss_pl_GB = perceptual_loss(self.real_B, self.fake_B, self.distorted_B, 
                                         self.mask_eyes_B, self.vggface_feats, **loss_weights)
        else:
            loss_pl_GA = loss_pl_GB = K.zeros(1)

        loss_GA = loss_adv_GA + loss_recon_GA + loss_edge_GA + loss_pl_GA
        loss_GB = loss_adv_GB + loss_recon_GB + loss_edge_GB + loss_pl_GB

        # The following losses are rather trivial, thus their wegihts are fixed.
        # Cycle consistency loss
        if loss_config['use_cyclic_loss']:
            loss_GA += 10 * cyclic_loss(self.netGA, self.netGB, self.real_A)
            loss_GB += 10 * cyclic_loss(self.netGB, self.netGA, self.real_B)

        # Alpha mask loss
        if not loss_config['use_mask_hinge_loss']:
            loss_GA += 1e-2 * K.mean(K.abs(self.mask_A))
            loss_GB += 1e-2 * K.mean(K.abs(self.mask_B))
        else:
            loss_GA += 0.1 * K.mean(K.maximum(0., loss_config['m_mask'] - self.mask_A))
            loss_GB += 0.1 * K.mean(K.maximum(0., loss_config['m_mask'] - self.mask_B))

        # Alpha mask total variation loss
        loss_GA += 0.1 * K.mean(first_order(self.mask_A, axis=1))
        loss_GA += 0.1 * K.mean(first_order(self.mask_A, axis=2))
        loss_GB += 0.1 * K.mean(first_order(self.mask_B, axis=1))
        loss_GB += 0.1 * K.mean(first_order(self.mask_B, axis=2))

        # L2 weight decay
        # https://github.com/keras-team/keras/issues/2662
        for loss_tensor in self.netGA.losses:
            loss_GA += loss_tensor
        for loss_tensor in self.netGB.losses:
            loss_GB += loss_tensor
        for loss_tensor in self.netDA.losses:
            loss_DA += loss_tensor
        for loss_tensor in self.netDB.losses:
            loss_DB += loss_tensor

        weightsDA = self.netDA.trainable_weights
        weightsGA = self.netGA.trainable_weights
        weightsDB = self.netDB.trainable_weights
        weightsGB = self.netGB.trainable_weights

        # Define training functions
        # Adam(...).get_updates(...)
        training_updates = Adam(lr=self.lrD*loss_config['lr_factor'], beta_1=0.5).get_updates(weightsDA,[],loss_DA)
        self.netDA_train = K.function([self.distorted_A, self.real_A],[loss_DA], training_updates)
        training_updates = Adam(lr=self.lrG*loss_config['lr_factor'], beta_1=0.5).get_updates(weightsGA,[], loss_GA)
        self.netGA_train = K.function([self.distorted_A, self.real_A, self.mask_eyes_A], 
                                      [loss_GA, loss_adv_GA, loss_recon_GA, loss_edge_GA, loss_pl_GA], 
                                      training_updates)

        training_updates = Adam(lr=self.lrD*loss_config['lr_factor'], beta_1=0.5).get_updates(weightsDB,[],loss_DB)
        self.netDB_train = K.function([self.distorted_B, self.real_B],[loss_DB], training_updates)
        training_updates = Adam(lr=self.lrG*loss_config['lr_factor'], beta_1=0.5).get_updates(weightsGB,[], loss_GB)
        self.netGB_train = K.function([self.distorted_B, self.real_B, self.mask_eyes_B], 
                                      [loss_GB, loss_adv_GB, loss_recon_GB, loss_edge_GB, loss_pl_GB], 
                                      training_updates)
    
    def build_pl_model(self, vggface_model):
        # Define Perceptual Loss Model
        vggface_model.trainable = False
        out_size112 = vggface_model.layers[1].output
        out_size55 = vggface_model.layers[36].output
        out_size28 = vggface_model.layers[78].output
        out_size7 = vggface_model.layers[-2].output
        self.vggface_feats = Model(vggface_model.input, [out_size112, out_size55, out_size28, out_size7])
        self.vggface_feats.trainable = False
    
    def load_weights(self, path="./models"):
        try:
            self.encoder.load_weights(f"{path}/encoder.h5")
            self.decoder_A.load_weights(f"{path}/decoder_A.h5")
            self.decoder_B.load_weights(f"{path}/decoder_B.h5")
            self.netDA.load_weights(f"{path}/netDA.h5") 
            self.netDB.load_weights(f"{path}/netDB.h5") 
            print ("Model weights files are successfully loaded")
        except:
            print ("Error occurs during loading weights files.")
            pass
        
    def train_one_batch_G(self, data_A, data_B):
        _, warped_A, target_A, bm_eyes_A = data_A
        _, warped_B, target_B, bm_eyes_B = data_B
        errGA = self.netGA_train([warped_A, target_A, bm_eyes_A])
        errGB = self.netGB_train([warped_B, target_B, bm_eyes_B])        
        return errGA, errGB
    
    def train_one_batch_D(self, data_A, data_B):
        _, warped_A, target_A, _ = data_A
        _, warped_B, target_B, _ = data_B
        errDA = self.netDA_train([warped_A, target_A])
        errDB = self.netDB_train([warped_B, target_B])
        return errDA, errDB
    
    def transform_A2B(self, img):
        return self.path_abgr_B([[ae_input]])
    
    def transform_B2A(self, img):
        return self.path_abgr_A([[ae_input]])