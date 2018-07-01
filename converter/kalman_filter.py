import cv2
import numpy as np

class KalmanFilter():
    def __init__(self, noise_coef):
        self.noise_coef = noise_coef
        self.kf = self.init_kalman_filter(noise_coef)
    
    @staticmethod
    def init_kalman_filter(noise_coef):
        kf = cv2.KalmanFilter(4,2)
        kf.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
        kf.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)
        kf.processNoiseCov = noise_coef * np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]], np.float32)
        return kf
    
    def correct(self, xy):
        return self.kf.correct(xy)
    
    def predict(self):
        return self.kf.predict()