from keras.engine import Layer, InputSpec
from keras import initializers, regularizers
from keras import backend as K
from keras.utils import conv_utils

try:
    from keras.utils.conv_utils import normalize_data_format
except:
    from keras.backend.common import normalize_data_format

def to_list(x):
    if type(x) not in [list, tuple]:
        return [x]
    else:
        return list(x)

class GroupNormalization(Layer):
    def __init__(self, axis=-1,
                 gamma_init='one', beta_init='zero',
                 gamma_regularizer=None, beta_regularizer=None,
                 epsilon=1e-6, 
                 group=32,
                 data_format=None,
                 **kwargs): 
        super(GroupNormalization, self).__init__(**kwargs)

        self.axis = to_list(axis)
        self.gamma_init = initializers.get(gamma_init)
        self.beta_init = initializers.get(beta_init)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.epsilon = epsilon
        self.group = group
        self.data_format = normalize_data_format(data_format)

        self.supports_masking = True

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        shape = [1 for _ in input_shape]       
        if self.data_format == 'channels_last':
            channel_axis = -1
            shape[channel_axis] = input_shape[channel_axis]
        elif self.data_format == 'channels_first':
            channel_axis = 1
            shape[channel_axis] = input_shape[channel_axis]
        #for i in self.axis:
        #    shape[i] = input_shape[i]
        self.gamma = self.add_weight(shape=shape,
                                     initializer=self.gamma_init,
                                     regularizer=self.gamma_regularizer,
                                     name='gamma')
        self.beta = self.add_weight(shape=shape,
                                    initializer=self.beta_init,
                                    regularizer=self.beta_regularizer,
                                    name='beta')
        self.built = True

    def call(self, inputs, mask=None):
        input_shape = K.int_shape(inputs)
        if len(input_shape) != 4 and len(input_shape) != 2:
            raise ValueError('Inputs should have rank ' +
                             str(4) + " or " + str(2) +
                             '; Received input shape:', str(input_shape))

        if len(input_shape) == 4:
            if self.data_format == 'channels_last':
                batch_size, h, w, c = input_shape
                if batch_size is None:
                    batch_size = -1
                
                if c < self.group:
                    raise ValueError('Input channels should be larger than group size' +
                                     '; Received input channels: ' + str(c) +
                                     '; Group size: ' + str(self.group)
                                    )

                x = K.reshape(inputs, (batch_size, h, w, self.group, c // self.group))
                mean = K.mean(x, axis=[1, 2, 4], keepdims=True)
                std = K.sqrt(K.var(x, axis=[1, 2, 4], keepdims=True) + self.epsilon)
                x = (x - mean) / std

                x = K.reshape(x, (batch_size, h, w, c))
                return self.gamma * x + self.beta
            elif self.data_format == 'channels_first':
                batch_size, c, h, w = input_shape
                if batch_size is None:
                    batch_size = -1
                
                if c < self.group:
                    raise ValueError('Input channels should be larger than group size' +
                                     '; Received input channels: ' + str(c) +
                                     '; Group size: ' + str(self.group)
                                    )

                x = K.reshape(inputs, (batch_size, self.group, c // self.group, h, w))
                mean = K.mean(x, axis=[2, 3, 4], keepdims=True)
                std = K.sqrt(K.var(x, axis=[2, 3, 4], keepdims=True) + self.epsilon)
                x = (x - mean) / std

                x = K.reshape(x, (batch_size, c, h, w))
                return self.gamma * x + self.beta
                
        elif len(input_shape) == 2:
            reduction_axes = list(range(0, len(input_shape)))
            del reduction_axes[0]
            batch_size, _ = input_shape
            if batch_size is None:
                batch_size = -1
                
            mean = K.mean(inputs, keepdims=True)
            std = K.sqrt(K.var(inputs, keepdims=True) + self.epsilon)
            x = (inputs  - mean) / std
            
            return self.gamma * x + self.beta
            

    def get_config(self):
        config = {'epsilon': self.epsilon,
                  'axis': self.axis,
                  'gamma_init': initializers.serialize(self.gamma_init),
                  'beta_init': initializers.serialize(self.beta_init),
                  'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
                  'beta_regularizer': regularizers.serialize(self.beta_regularizer),
                  'group': self.group
                 }
        base_config = super(GroupNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
