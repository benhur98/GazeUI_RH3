import winsound as sound
from get_key import k_check
from imutils import face_utils
import os
import imutils
import cv2
import time
from Camera import camera
import dlib
import numpy as  np
from Feature import contour

FILE_NAME="FACIAL_LM.dat"
dl=dlib.shape_predictor(FILE_NAME)
##incase of url-based-ip stream ipwebcam app
##urlstream="http://192.168.1.2:8080/video"
cam=camera.VideoFeed()
cntr=contour.Contour(dl)
face=camera.Face()
detector=face.cam()
train_data=[]
(LS, LE) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"] 
(RS, RE) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

##keys to store
a = [1,0,0,0,0,0,0,0,0]
b = [0,1,0,0,0,0,0,0,0]
c = [0,0,1,0,0,0,0,0,0]
d = [0,0,0,1,0,0,0,0,0]
e = [0,0,0,0,1,0,0,0,0]
f = [0,0,0,0,0,1,0,0,0]
g = [0,0,0,0,0,0,1,0,0]
h = [0,0,0,0,0,0,0,1,0]
nk =[0,0,0,0,0,0,0,0,1]

start_value = 1
while True:
    file_name = 'train-data-{}.npy'.format(start_value)

    if os.path.isfile(file_name):
        print('File already present, looping - ', start_value)
        start_value += 1
    else:
        print('File exist found null!, starting fresh!', start_value)
        break

def keys_output(keys):
    '''
    Convert key vals to multi-hot array..
     0  1  2  3  4   5   6   7    8
    [a, b, c, d, e, f, g, h, NOKEY] bool values.
    '''
    output = [0,0,0,0,0,0,0,0,0]
#changed
    if '1' in keys:
        output = b
    elif '2' in keys:
        output = c
    elif 'c' in keys:
        output = b
    elif 'v' in keys:
        output = d
    elif '5' in keys:
        output = e
    elif '6' in keys:
        output = f
    elif '7' in keys:
        output = g
    elif '8' in keys:
        output = h
    else:
        output = a
    return output
print("collecting data's for training in --/")
for i in list(range(4))[::-1]:
    print(i+1)
    time.sleep(.5)
    
while 1:
    frame=cam.getframe()
    frame_G=cam.getframe(True)
    frame_2=frame.copy()

    coordpts=detector(frame,0)
    coordpts=cntr.dots(frame,coordpts)
    if coordpts is not None:
        for point in coordpts:
            cv2.circle(frame, tuple(point), 1, (0, 0, 255))
        lefteye=coordpts[LS:LE]
        righteye=coordpts[RS:RE]
        cv2.circle(frame, (int((righteye[0][0]+righteye[3][0])/2),int((righteye[0][1]+righteye[3][1])/2)), 1, (0, 0, 244), -1)
        cv2.circle(frame, (int((lefteye[0][0]+lefteye[3][0])/2),int((lefteye[0][1]+lefteye[3][1])/2)), 1, (0, 0, 244), -1)
        (x,y,w,h)=cv2.boundingRect(np.array([lefteye]))
        (xr, yr, wr, hr) = cv2.boundingRect(np.array([righteye]))
        off=10
        roi_l = imutils.resize(frame_2[y-off:y+off+h+off,x-off:x+w+off], width=250, inter=cv2.INTER_CUBIC)
        roi_r = imutils.resize(frame_2[yr-off:yr+off+hr+off,xr-off:xr+wr+off], width=250, inter=cv2.INTER_CUBIC)
        roi_l_resized=cv2.resize(roi_l, (250, 250))
        roi_r_resized = cv2.resize(roi_r, (250, 250))
        roi_l_resized = cv2.cvtColor(roi_l_resized, cv2.COLOR_BGR2RGB)
        roi_r_resized = cv2.cvtColor(roi_r_resized, cv2.COLOR_BGR2RGB)
##manual routine  to reroute keys to data gen
##change output key cycle

        keys_train=k_check()
        output_train=keys_output(keys_train)

        #creation of training data
        train_data.append([roi_r_resized,output_train])
        if len(train_data) % 100 == 0:
            print(len(train_data))
            #sound alert %10
            #sound.Beep(420,200)

            if len(train_data) == 200:
                np.save(file_name, train_data)
                print('-SAVED-')
                train_data = []
                start_value += 1
                file_name = 'train-data-{}.npy'.format(start_value)
                break

    k=cv2.waitKey(23) & 0xFF
    if k is ord('q'):
        cv2.destroyAllWindows()
        break

if __name__=="__main__":
    pass














    

