import numpy as np
import cv2
from umeyama import umeyama
from scipy import ndimage
from pathlib import PurePath, Path

random_transform_args = {
    'rotation_range': 10,
    'zoom_range': 0.1,
    'shift_range': 0.05,
    'random_flip': 0.5,
    }

# Motion blurs as data augmentation
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

def random_transform(image, rotation_range, zoom_range, shift_range, random_flip):
    h,w = image.shape[0:2]
    rotation = np.random.uniform(-rotation_range, rotation_range)
    scale = np.random.uniform(1 - zoom_range, 1 + zoom_range)
    tx = np.random.uniform(-shift_range, shift_range) * w
    ty = np.random.uniform(-shift_range, shift_range) * h
    mat = cv2.getRotationMatrix2D((w//2,h//2), rotation, scale)
    mat[:,2] += (tx,ty)
    result = cv2.warpAffine(image, mat, (w,h), borderMode=cv2.BORDER_REPLICATE)
    if np.random.random() < random_flip:
        result = result[:,::-1]
    return result

def random_warp_rev(image, res=64):
    assert image.shape == (256,256,6)
    res_scale = res//64
    assert res_scale >= 1, f"Resolution should be >= 64. Recieved {res}."
    interp_param = 80 * res_scale
    interp_slice = slice(interp_param//10,9*interp_param//10)
    dst_pnts_slice = slice(0,65*res_scale,16*res_scale)
    
    rand_coverage = np.random.randint(20) + 78 # random warping coverage
    rand_scale = np.random.uniform(5., 6.2) # random warping scale
    
    range_ = np.linspace(128-rand_coverage, 128+rand_coverage, 5)
    mapx = np.broadcast_to(range_, (5,5))
    mapy = mapx.T
    mapx = mapx + np.random.normal(size=(5,5), scale=rand_scale)
    mapy = mapy + np.random.normal(size=(5,5), scale=rand_scale)
    interp_mapx = cv2.resize(mapx, (interp_param,interp_param))[interp_slice,interp_slice].astype('float32')
    interp_mapy = cv2.resize(mapy, (interp_param,interp_param))[interp_slice,interp_slice].astype('float32')
    warped_image = cv2.remap(image, interp_mapx, interp_mapy, cv2.INTER_LINEAR)
    src_points = np.stack([mapx.ravel(), mapy.ravel()], axis=-1)
    dst_points = np.mgrid[dst_pnts_slice,dst_pnts_slice].T.reshape(-1,2)
    mat = umeyama(src_points, dst_points, True)[0:2]
    target_image = cv2.warpAffine(image, mat, (res,res))
    return warped_image, target_image

def random_color_match(image, fns_all_trn_data):
    rand_idx = np.random.randint(len(fns_all_trn_data))    
    fn_match = fns_all_trn_data[rand_idx]
    tar_img = cv2.imread(fn_match)
    if tar_img is None:
        print(f"Failed reading image {fn_match} in random_color_match().")
        return image
    r = 60 # only take color information of the center area
    src_img = cv2.resize(image, (256,256))
    tar_img = cv2.resize(tar_img, (256,256))  
    
    # randomly transform to XYZ color space
    rand_color_space_to_XYZ = np.random.choice([True, False])
    if rand_color_space_to_XYZ:
        src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2XYZ)
        tar_img = cv2.cvtColor(tar_img, cv2.COLOR_BGR2XYZ)
    
    # compute statistics
    mt = np.mean(tar_img[r:-r,r:-r,:], axis=(0,1))
    st = np.std(tar_img[r:-r,r:-r,:], axis=(0,1))
    ms = np.mean(src_img[r:-r,r:-r,:], axis=(0,1))
    ss = np.std(src_img[r:-r,r:-r,:], axis=(0,1))    
    
    # randomly interpolate the statistics
    rand_ratio = np.random.uniform()
    mt = rand_ratio * mt + (1 - rand_ratio) * ms
    st = rand_ratio * st + (1 - rand_ratio) * ss
    
    # Apply color transfer from src to tar domain
    if ss.any() <= 1e-7: return src_img    
    result = st * (src_img.astype(np.float32) - ms) / (ss+1e-7) + mt
    if result.min() < 0:
        result = result - result.min()
    if result.max() > 255:
        result = (255.0/result.max()*result).astype(np.float32)
    
    # transform back from XYZ to BGR color space if necessary
    if rand_color_space_to_XYZ:
        result = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_XYZ2BGR)
    return result

def read_image(fn, fns_all_trn_data, dir_bm_eyes=None, res=64, prob_random_color_match=0.5, 
               use_da_motion_blur=True, use_bm_eyes=True, 
               random_transform_args=random_transform_args):
    if dir_bm_eyes is None:
        raise ValueError(f"dir_bm_eyes is not set.")
        
    # https://github.com/tensorflow/tensorflow/issues/5552
    # TensorFlow converts str to bytes in most places, including sess.run().
    if type(fn) == type(b"bytes"):
        fn = fn.decode("utf-8")
        dir_bm_eyes = dir_bm_eyes.decode("utf-8")
        fns_all_trn_data = [fn_all.decode("utf-8") for fn_all in fns_all_trn_data]
    
    raw_fn = PurePath(fn).parts[-1]
    image = cv2.imread(fn)
    if image is None:
        print(f"Failed reading image {fn}.")
        raise IOError(f"Failed reading image {fn}.")        
    if np.random.uniform() <= prob_random_color_match:
        image = random_color_match(image, fns_all_trn_data)
    image = cv2.resize(image, (256,256)) / 255 * 2 - 1
    
    if use_bm_eyes:
        bm_eyes = cv2.imread(f"{dir_bm_eyes}/{raw_fn}")
        if bm_eyes is None:
            print(f"Failed reading binary mask {dir_bm_eyes}/{raw_fn}. \
            If this message keeps showing, please check for existence of binary masks folder \
            or disable eye-aware training in the configuration.")
            bm_eyes = np.zeros_like(image)
            #raise IOError(f"Failed reading binary mask {dir_bm_eyes}/{raw_fn}.")
        bm_eyes = cv2.resize(bm_eyes, (256,256)) / 255.
    else:
        bm_eyes = np.zeros_like(image)
    
    image = np.concatenate([image, bm_eyes], axis=-1)
    image = random_transform(image, **random_transform_args)
    warped_img, target_img = random_warp_rev(image, res=res)
    
    bm_eyes = target_img[...,3:]
    warped_img = warped_img[...,:3]
    target_img = target_img[...,:3]
    
    # Motion blur data augmentation:
    # we want the model to learn to preserve motion blurs of input images
    if np.random.uniform() < 0.25 and use_da_motion_blur: 
        warped_img, target_img = motion_blur([warped_img, target_img])
    
    warped_img, target_img, bm_eyes = \
    warped_img.astype(np.float32), target_img.astype(np.float32), bm_eyes.astype(np.float32)
    
    return warped_img, target_img, bm_eyes