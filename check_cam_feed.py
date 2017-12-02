from imutils import face_utils
from scipy.spatial import distance as dist
import cv2
from Camera import camera
import imutils
import dlib
import numpy as np
import os
from Feature import contour
print(os.getcwd())
FN="FACIAL_LM.dat"
dl=dlib.shape_predictor(FN)
#urlstream="http://192.168.1.2:8080/video"
cam=camera.VideoFeed()
contour=contour.Contour(dl)
face=camera.Face()
detecor=face.cam()
methods = [
	 cv2.THRESH_BINARY,
	 cv2.THRESH_BINARY_INV,
	 cv2.THRESH_TRUNC,
     cv2.THRESH_TOZERO,
	 cv2.THRESH_TOZERO_INV]

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)

    return ear
def draw_rectangle2(image, coords):
    for rect in coords:
        print(rect)
        a=np.asarray(rect)
        cv2.rectangle(image, a[0][0],y[0][1],(76, 231, 29), 3)


def draw_rectangle(image, coords):
    for (x, y, w, h) in coords:
        w_rm = int(0.2 * w / 2)
        cv2.rectangle(image, (x + w_rm, y), (x + w - w_rm, y + h),
                      (76, 231, 29), 4)
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

while 1:
    frame=cam.getframe()
    frame_2 = frame.copy()
    coordpts=detecor(frame,0)
    coordpts=contour.dots(frame,coordpts)
    if coordpts is not None:
        for point in coordpts:
            cv2.circle(frame, tuple(point), 1, (0, 0, 255))
        lefteye=coordpts[lStart:lEnd]
        righteye=coordpts[rStart:rEnd]
        lefthull=cv2.convexHull(lefteye)
        righthull=cv2.convexHull(righteye)

        cv2.circle(frame, (int((righteye[0][0]+righteye[3][0])/2),int((righteye[0][1]+righteye[3][1])/2)), 1, (0, 0, 244), -1)
        cv2.circle(frame, (int((lefteye[0][0]+lefteye[3][0])/2),int((lefteye[0][1]+lefteye[3][1])/2)), 1, (0, 0, 244), -1)
        eye_ratio=(eye_aspect_ratio(lefteye)+ eye_aspect_ratio(righteye)) /2
        (x,y,w,h)=cv2.boundingRect(np.array([lefteye]))
        (xr, yr, wr, hr) = cv2.boundingRect(np.array([righteye]))
        off=10
        roi_l = imutils.resize(frame_2[y-off:y+off+h+off,x-off:x+w+off], width=250, inter=cv2.INTER_CUBIC)
        roi_r = imutils.resize(frame_2[yr-off:yr+off+hr+off,xr-off:xr+wr+off], width=250, inter=cv2.INTER_CUBIC)
        roi_l_g=cv2.cvtColor(roi_l,cv2.COLOR_BGR2GRAY)
        roi_r_g = cv2.cvtColor(roi_l, cv2.COLOR_BGR2GRAY)
        # blurred_l = cv2.GaussianBlur(roi_l_g, (7, 7), 0)
        # blurred_r = cv2.GaussianBlur(roi_r_g, (7, 7), 0)
        # blurred_l = cv2.medianBlur(roi_l_g,  5)
        # blurred_r = cv2.medianBlur(roi_r_g,  5)
        # th3 = cv2.adaptiveThreshold(blurred_l, 255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY, 11, 2)
        # edged_l = cv2.Canny(blurred_l, 100, 300)
        # edged_r = cv2.Canny(blurred_r, 100, 300)
        #(T, thresh) = cv2.threshold(blurred_l, 100, 100, methods[1])
        #
        # _,cnts_l,_= cv2.findContours(edged_l.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        # _,cnts_r,_ = cv2.findContours(edged_r.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        #
        # _,th3 = cv2.threshold(blurred_l,127, 255, cv2.THRESH_BINARY)
        # _, cnts_l, _ = cv2.findContours(th3.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # # print(cnts_l)
        # cv2.drawContours(roi_l, cnts_l, -1, (0, 0, 255), 4)
        # cv2.drawContours(roi_r, cnts_r, -1, (0, 0, 255), 4)
        #------------------------------------------
       # circles = cv2.HoughCircles(blurred_l, cv2.HOUGH_GRADIENT, 1, 120, param1=100, param2=100, minRadius=0, maxRadius=0)

        # circles = np.uint16(np.around(circles))
        #besttttttttttttttt if bg light dot
        # (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(roi_l_g)
        # cv2.circle(roi_l, maxLoc, 10, (255, 0, 0))

        #
        # if cnts_l != None:
        #     for cnts in cnts_l:
        #         if cnts !=None:
        #             if len(cnts)<20:
        #                 print(cnts)
        #
        #                 h=cv2.convexHull(cnts)
        #
        #                 cv2.drawContours(roi_l, [h], -1, (0, 0, 255), 4)
                        #
                        # peri = cv2.arcLength(cnts_l, True)
                        # approx = cv2.approxPolyDP(cnts_l, 0.01 * peri, True)
                        # cv2.drawContours(roi_l, [approx], -1, (0, 0, 255), 4)


        # circles = cv2.HoughCircles(edged_l, cv2.HOUGH_GRADIENT, 1.2, 20)
        # if circles != None:
        #     circles = np.uint16(np.around(circles))
        #     for i in circles[0, :]:
        #         # draw the outer circle
        #         cv2.circle(roi_l, (i[0], i[1]), i[2], (0, 255, 0), 2)
        #         # draw the center of the circle
        #         cv2.circle(roi_l, (i[0], i[1]), 2, (0, 0, 255), 3)
        #
        #clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(2, 2))
        #cl1 = clahe.apply(roi_l_g)
        cv2.imshow("live feed - 1", roi_l)
        cv2.imshow("live feed - 2", roi_r)



        #print(eye_ratio)
    #draw_rectangle2(frame,coords)


    #edged = cv2.Canny(framegray, 50, 150)

    cv2.imshow("live feed",frame)



    k=cv2.waitKey(23) & 0xFF
    if k is ord('q'):
        cv2.destroyAllWindows()
        break






















if __name__=="__main__":
    pass
