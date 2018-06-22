from IPython.display import display
from PIL import Image
import cv2
import numpy as np
import os

def get_image_paths(directory):
    return [x.path for x in os.scandir(directory) if x.name.endswith(".jpg") or x.name.endswith(".png")]

def load_images(image_paths, convert=None):
    iter_all_images = (cv2.resize(cv2.imread(fn), (256,256)) for fn in image_paths)
    if convert:
        iter_all_images = (convert(img) for img in iter_all_images)
    for i,image in enumerate( iter_all_images ):
        if i == 0:
            all_images = np.empty((len(image_paths),) + image.shape, dtype=image.dtype)
        all_images[i] = image
    return all_images

def get_transpose_axes( n ):
    if n % 2 == 0:
        y_axes = list(range(1, n-1, 2))
        x_axes = list(range(0, n-1, 2))
    else:
        y_axes = list(range(0, n-1, 2))
        x_axes = list(range(1, n-1, 2))
    return y_axes, x_axes, [n-1]

def stack_images(images):
    images_shape = np.array(images.shape)
    new_axes = get_transpose_axes(len(images_shape))
    new_shape = [np.prod(images_shape[x]) for x in new_axes]
    return np.transpose(
        images,
        axes = np.concatenate(new_axes)
        ).reshape(new_shape)

def showG(test_A, test_B, path_A, path_B, batchSize):
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

    figure = np.concatenate([figure_A, figure_B], axis=0)
    figure = figure.reshape((4,batchSize//2) + figure.shape[1:])
    figure = stack_images(figure)
    figure = np.clip((figure + 1) * 255 / 2, 0, 255).astype('uint8')
    figure = cv2.cvtColor(figure, cv2.COLOR_BGR2RGB)

    display(Image.fromarray(figure))
    
def showG_mask(test_A, test_B, path_A, path_B, batchSize):
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

    figure = np.concatenate([figure_A, figure_B], axis=0)
    figure = figure.reshape((4,batchSize//2) + figure.shape[1:])
    figure = stack_images(figure)
    figure = np.clip((figure + 1) * 255 / 2, 0, 255).astype('uint8')
    figure = cv2.cvtColor(figure, cv2.COLOR_BGR2RGB)

    display(Image.fromarray(figure))
    
def showG_eyes(test_A, test_B, bm_eyes_A, bm_eyes_B, batchSize):
    figure_A = np.stack([
        (test_A + 1)/2,
        bm_eyes_A,
        bm_eyes_A * (test_A + 1)/2,
        ], axis=1 )
    figure_B = np.stack([
        (test_B + 1)/2,
        bm_eyes_B,
        bm_eyes_B * (test_B+1)/2,
        ], axis=1 )

    figure = np.concatenate([figure_A, figure_B], axis=0)
    figure = figure.reshape((4,batchSize//2) + figure.shape[1:])
    figure = stack_images(figure)
    figure = np.clip(figure * 255, 0, 255).astype('uint8')
    figure = cv2.cvtColor(figure, cv2.COLOR_BGR2RGB)

    display(Image.fromarray(figure))
