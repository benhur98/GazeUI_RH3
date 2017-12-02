import imp_1
import pyautogui
from imutils import face_utils
from scipy.spatial import distance as dist
import cv2
from Camera import camera
import imutils
import dlib
import numpy as np
import time
import os
from get_key import k_check
from Feature import contour
from modelsNN import inceptionv3 as gnet

LR = 1e-3
WIDTH = 250
from threading import Thread
HEIGHT = 250
i=1
MODEL_NAME="CNN-{}".format(i)
LOAD_MODEL = True
count=0
a = [1,0,0,0,0,0,0,0,0]
b = [0,1,0,0,0,0,0,0,0]
c = [0,0,1,0,0,0,0,0,0]
d = [0,0,0,1,0,0,0,0,0]
e = [0,0,0,0,1,0,0,0,0]
f = [0,0,0,0,0,1,0,0,0]
g = [0,0,0,0,0,0,1,0,0]
h = [0,0,0,0,0,0,0,1,0]
nk = [0,0,0,0,0,0,0,0,1]
FN="FACIAL_LM.dat"
dl=dlib.shape_predictor(FN)
#urlstream="http://192.168.1.2:8080/video"
cam=camera.VideoFeed()
contour=contour.Contour(dl)
face=camera.Face()
COUNTER = 0
detecor=face.cam()
EYE_THR = 0.3
EYE_CLOSE_FRAMES = 5
train_data=[]
STATE_CLICK=False
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
model = gnet(WIDTH, HEIGHT, 3, LR, output=9, model_name=MODEL_NAME)

if LOAD_MODEL:
    model.load(MODEL_NAME)

    print('loaded a previous model into inception_v3!!')
for i in list(range(4))[::-1]:
    print(i+1)
    time.sleep(.1)
print("Started..")
state_paused=False

COUNTER=0
def click():
    pyautogui.click()
    
def click_mouse_action():
        global COUNTER
        global STATE_CLICK

        if eye_ratio < EYE_THR:
            COUNTER=COUNTER+1
            if COUNTER > EYE_CLOSE_FRAMES:
                if not STATE_CLICK:
                    STATE_CLICK = True
                    t = Thread(target=click,)
                    t.deamon = True
                    t.start()
        else:
            COUNTER = 0
            STATE_CLICK = False
def choice2():
    
    click_mouse_action()
    print("looked R8 !! action took.....")
def choice1():
    click_mouse_action()
    print("looked left !! action took.....")

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)

    return ear    
while 1 and not state_paused:
    frame=cam.getframe()
    frame_2 = frame.copy()
    framegray=cam.getframe(True)


    coordpts=detecor(frame,0)
    coordpts=contour.dots(frame,coordpts)
    if coordpts is not None:
        for point in coordpts:
            cv2.circle(frame, tuple(point), 1, (0, 0, 255))
        lefteye=coordpts[lStart:lEnd]
        righteye=coordpts[rStart:rEnd]

        lefteye=coordpts[lStart:lEnd]
        righteye=coordpts[rStart:rEnd]
        lefthull=cv2.convexHull(lefteye)
        righthull=cv2.convexHull(righteye)
        eye_ratio=eye_aspect_ratio(lefteye)




        
        cv2.circle(frame, (int((righteye[0][0]+righteye[3][0])/2),int((righteye[0][1]+righteye[3][1])/2)), 1, (0, 0, 244), -1)
        cv2.circle(frame, (int((lefteye[0][0]+lefteye[3][0])/2),int((lefteye[0][1]+lefteye[3][1])/2)), 1, (0, 0, 244), -1)
        (x,y,w,h)=cv2.boundingRect(np.array([lefteye]))
        (xr, yr, wr, hr) = cv2.boundingRect(np.array([righteye]))
        off=10
        roiL = imutils.resize(frame[y - off:y + off + h + off, x - off:x + w + off], width=250, height=250,
                               inter=cv2.INTER_CUBIC)
        roiR = imutils.resize(frame[yr - off:yr + off + hr + off, xr - off:xr + wr + off], width=250, height=250,
                               inter=cv2.INTER_CUBIC)
        roiL = cv2.resize(roiL, (250, 250))
        roiR = cv2.resize(roiR, (250, 250))


        roi_l = imutils.resize(frame_2[y-off:y+off+h+off,x-off:x+w+off], width=250, height=250,inter=cv2.INTER_CUBIC)
        roi_r = imutils.resize(frame_2[yr-off:yr+off+hr+off,xr-off:xr+wr+off], width=250, height=250,inter=cv2.INTER_CUBIC)
        roi_l_resized=cv2.resize(roi_l, (250, 250))
        roi_r_resized = cv2.resize(roi_r, (250, 250))
        # roi_lo=roi_l_resized
        # roi_ro=roi_r_resized
        roi_l_resized = cv2.cvtColor(roi_l_resized, cv2.COLOR_BGR2RGB)
        roi_r_resized = cv2.cvtColor(roi_r_resized, cv2.COLOR_BGR2RGB)

        if count>5:
            
            prediction = model.predict([roi_r_resized.reshape(WIDTH,HEIGHT,3)])
            prediction2 = model2.model2.predict([roi_r_resized.reshape(WIDTH, HEIGHT, 3)])
            

            # np.round(prediction)
            # np.round(prediction2)
            
            choice_pred_index = np.argmax(prediction)
            choice_pred_index2 = np.argmax(prediction2)
            print(choice_pred_index," -->new ",choice_pred_index2)
            if choice_pred_index == 0 and choice_pred_index2==0:
##                pyautogui.moveTo(598,570)
                #pass
                #choice1()
                pass
            elif choice_pred_index2 == 1:
##                pyautogui.moveTo(196,586)
                choice1()

                pass
            elif choice_pred_index == 4:
##                pyautogui.moveTo(1203,597)
                choice2()
                pass
            else:
                pass



            cv2.imshow("live feed - 1", roiL)
            cv2.imshow("live feed - 2", roiR)
            
        count=count+1
    cv2.imshow("live feed",frame)
    keys = k_check()

        # p pauses game and can get annoying.
    if 'p' in keys:
        if not state_paused:
            state_paused = True
            time.sleep(1)
        else:
            state_paused=False
            time.sleep(1)

    k=cv2.waitKey(23) & 0xFF
    if k is ord('q'):
        cv2.destroyAllWindows()
        break

if __name__=="__main__":
    pass
