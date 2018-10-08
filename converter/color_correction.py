import numpy as np
import cv2

""" Color corretion functions"""
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

def color_hist_match(src_im, tar_im, color_space="RGB"):
    if color_space.lower() != "rgb":
        src_im = trans_color_space(src_im, color_space)
        tar_im = trans_color_space(tar_im, color_space)
        
    matched_R = hist_match(src_im[:,:,0], tar_im[:,:,0])
    matched_G = hist_match(src_im[:,:,1], tar_im[:,:,1])
    matched_B = hist_match(src_im[:,:,2], tar_im[:,:,2])
    matched = np.stack((matched_R, matched_G, matched_B), axis=2).astype(np.float32)
    matched = np.clip(matched, 0, 255)
    
    if color_space.lower() != "rgb":
        result = trans_color_space(result.astype(np.uint8), color_space, rev=True)
    return matched

def adain(src_im, tar_im, eps=1e-7, color_space="RGB"):
    # https://github.com/ftokarev/tf-adain/blob/master/adain/norm.py
    if color_space.lower() != "rgb":
        src_im = trans_color_space(src_im, color_space)
        tar_im = trans_color_space(tar_im, color_space)
        
    mt = np.mean(tar_im, axis=(0,1))
    st = np.std(tar_im, axis=(0,1))
    ms = np.mean(src_im, axis=(0,1))
    ss = np.std(src_im, axis=(0,1))    
    if ss.any() <= eps: return src_im    
    result = st * (src_im.astype(np.float32) - ms) / (ss+eps) + mt
    result = np.clip(result, 0, 255)  
        
    if color_space.lower() != "rgb":
        result = trans_color_space(result.astype(np.uint8), color_space, rev=True)
    return result

def trans_color_space(im, color_space, rev=False):
    if color_space.lower() == "lab":
        clr_spc = cv2.COLOR_BGR2Lab
        rev_clr_spc = cv2.COLOR_Lab2BGR
    elif color_space.lower() == "ycbcr":
        clr_spc = cv2.COLOR_BGR2YCR_CB
        rev_clr_spc = cv2.COLOR_YCR_CB2BGR
    elif color_space.lower() == "xyz":
        clr_spc = cv2.COLOR_BGR2XYZ
        rev_clr_spc = cv2.COLOR_XYZ2BGR
    elif color_space.lower() == "luv":
        clr_spc = cv2.COLOR_BGR2Luv
        rev_clr_spc = cv2.COLOR_Luv2BGR
    elif color_space.lower() == "rgb":
        pass
    else:
        raise NotImplementedError()
        
    if color_space.lower() != "rgb":
        trans_clr_spc = rev_clr_spc if rev else clr_spc
        im = cv2.cvtColor(im, trans_clr_spc)
    return im
    