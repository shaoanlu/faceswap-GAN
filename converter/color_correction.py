import numpy as np

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

def color_hist_match(src_im, tar_im):
    #src_im = cv2.cvtColor(src_im, cv2.COLOR_BGR2Lab)
    #tar_im = cv2.cvtColor(tar_im, cv2.COLOR_BGR2Lab)
    matched_R = hist_match(src_im[:,:,0], tar_im[:,:,0])
    matched_G = hist_match(src_im[:,:,1], tar_im[:,:,1])
    matched_B = hist_match(src_im[:,:,2], tar_im[:,:,2])
    matched = np.stack((matched_R, matched_G, matched_B), axis=2).astype(np.float32)
    return matched

def adain(src_im, tar_im, eps=1e-7):
    # https://github.com/ftokarev/tf-adain/blob/master/adain/norm.py
    mt = np.mean(tar_im, axis=(0,1))
    st = np.std(tar_im, axis=(0,1))
    ms = np.mean(src_im, axis=(0,1))
    ss = np.std(src_im, axis=(0,1))    
    if ss.any() <= eps: return src_im    
    result = st * (src_im.astype(np.float32) - ms) / (ss+eps) + mt
    if result.min() < 0:
        result = result - result.min()
    if result.max() > 255:
        result = (255.0/result.max()*result).astype(np.float32)
    return result