from scipy.spatial import distance as dist
from imutils import face_utils
import cv2
import numpy as np
import dlib


(LS,LE)=face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(RS,RE)=face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
class Contour(object):
    def __init__(self,predictor):
        self.EYE_THR=0.3
        self.EYE_FRAMES=10
        self.predictor=predictor
        
    def dots(self,image_g,rect_v):
        for rect in rect_v:
            shape=self.predictor(image_g,rect)
            shape=face_utils.shape_to_np(shape)
            return shape
    def L_EYE_vals(self,shape):

        return shape[LS:LE],shape[RS:RE]
    def EAR(self,eye):
        a=dist.euclidean(eye[1],eye[5])
        b=dist.euclidean(eye[2],eye[4])
        c=dist.euclidean(eye[0],eye[3])
        ear=(a+b)/(2.0* c)
        return ear
    def draweyes(self,frame,leftEye,rightEye):
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        


        
        
        
        
