import tensorflow as tf

def icnr_keras(shape, dtype=None):
    """
    From https://github.com/kostyaev/ICNR      
    Custom initializer for subpix upscaling
    Note: upscale factor is fixzed to 2, and the base initializer is fixed to random normal.
    """
    shape = list(shape)
    
    scale = 2
    initializer = tf.keras.initializers.RandomNormal(0, 0.02)

    new_shape = shape[:3] + [int(shape[3] / (scale ** 2))]
    x = initializer(new_shape, dtype)
    x = tf.transpose(x, perm=[2, 0, 1, 3])
    x = tf.image.resize_nearest_neighbor(x, size=(shape[0] * scale, shape[1] * scale))
    x = tf.space_to_depth(x, block_size=scale)
    x = tf.transpose(x, perm=[1, 2, 0, 3])
    return x
