import keras.backend as K
from moviepy.editor import VideoFileClip
from matplotlib import pyplot as plt
from pathlib import Path
import os

from converter.landmarks_alignment import *

class VideoInfo:
    def __init__(self):
        self.frame = 0

def process_image(input_img, info, detector, save_interval, save_path): 
    minsize = 30 # minimum size of face
    detec_threshold = 0.9
    threshold = [0.7, 0.8, detec_threshold]  # three steps's threshold
    factor = 0.709 # scale factor   
    
    info.frame += 1
    frame = info.frame
    if frame % save_interval == 0:
        faces, pnts = detector.detect_face(input_img, threshold=detec_threshold, use_auto_downscaling=False)
        for idx, (x0, y1, x1, y0, conf_score) in enumerate(faces):
            det_face_im = input_img[int(x0):int(x1),int(y0):int(y1),:]

            # get src/tar landmarks
            src_landmarks = get_src_landmarks(x0, x1, y0, y1, pnts)
            tar_landmarks = get_tar_landmarks(det_face_im)

            # align detected face
            aligned_det_face_im = landmarks_match_mtcnn(det_face_im, src_landmarks, tar_landmarks)

            Path(os.path.join(f"{save_path}", "rgb")).mkdir(parents=True, exist_ok=True)
            fname = os.path.join(f"{save_path}", "rgb", f"frame{frame}face{str(idx)}.jpg")
            plt.imsave(fname, aligned_det_face_im, format="jpg")
            #fname = f"./faces/raw_faces/frame{frames}face{str(idx)}.jpg"
            #plt.imsave(fname, det_face_im, format="jpg")
            
            bm = np.zeros_like(aligned_det_face_im)
            h, w = bm.shape[:2]
            bm[int(src_landmarks[0][0]-h/15):int(src_landmarks[0][0]+h/15),
               int(src_landmarks[0][1]-w/8):int(src_landmarks[0][1]+w/8),:] = 255
            bm[int(src_landmarks[1][0]-h/15):int(src_landmarks[1][0]+h/15),
               int(src_landmarks[1][1]-w/8):int(src_landmarks[1][1]+w/8),:] = 255
            bm = landmarks_match_mtcnn(bm, src_landmarks, tar_landmarks)
            Path(os.path.join(f"{save_path}", "binary_mask")).mkdir(parents=True, exist_ok=True)
            fname = os.path.join(f"{save_path}", "binary_mask", f"frame{frame}face{str(idx)}.jpg")
            plt.imsave(fname, bm, format="jpg")
        
    return np.zeros((3,3,3))

def preprocess_video(fn_input_video, fd, save_interval, save_path):
    info = VideoInfo()
    output = 'dummy.mp4'
    clip1 = VideoFileClip(fn_input_video)
    clip = clip1.fl_image(lambda img: process_image(img, info, fd, save_interval, save_path))
    clip.write_videofile(output, audio=False, verbose=False)
    clip1.reader.close()

        
