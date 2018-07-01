from umeyama import umeyama
import numpy as np
import cv2

def get_src_landmarks(x0, x1, y0, y1, pnts):
    """
    x0, x1, y0, y1: (smoothed) bbox coord.
    pnts: landmarks predicted by MTCNN
    """    
    src_landmarks = [(int(pnts[i+5][0]-x0), int(pnts[i][0]-y0)) for i in range(5)]
    return src_landmarks

def get_tar_landmarks(img):
    """    
    img: detected face image
    """         
    avg_landmarks = [
        (0.31339227236234224, 0.3259269274198092),
        (0.31075140146108776, 0.7228453709528997),
        (0.5523683107816256, 0.5187296867370605),
        (0.7752419985257663, 0.37262483743520886),
        (0.7759613623985877, 0.6772957581740159)
        ]       
    img_sz = img.shape
    tar_landmarks = [(int(xy[0]*img_sz[0]), int(xy[1]*img_sz[1])) for xy in avg_landmarks]
    return tar_landmarks

def landmarks_match_mtcnn(src_im, src_landmarks, tar_landmarks): 
    """
    umeyama(src, dst, estimate_scale), 
    src/dst landmarks coord. should be (y, x)
    """
    src_size = src_im.shape
    src_tmp = [(int(xy[1]), int(xy[0])) for xy in src_landmarks]
    dst_tmp = [(int(xy[1]), int(xy[0])) for xy in tar_landmarks]
    M = umeyama(np.array(src_tmp), np.array(dst_tmp), True)[0:2]
    result = cv2.warpAffine(src_im, M, (src_size[1], src_size[0]), borderMode=cv2.BORDER_REPLICATE) 
    return result