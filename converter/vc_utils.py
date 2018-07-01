import numpy as np
import cv2

# ==================================================
# Output image initialization functions
# ==================================================    
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