import cv2
import imutils
import dlib
class Face(object):
    def __init__(self,xml_path='frontal-face.xml'):
        self.classifier=cv2.CascadeClassifier(xml_path)
    def cam(self):
        detector=dlib.get_frontal_face_detector()
        return detector



    def detect_face(self,image):
        scale=1.2
        min_neigh=5
        min_size=(30,30)
        flags=12
        face=self.classifier.detectMultiScale(image,scaleFactor=scale,minNeighbors=min_neigh,minSize=min_size,flags=flags)

        return face


    
class VideoFeed(object):
    def __init__(self,index=1):
        self.video=cv2.VideoCapture(index)
        self.index=index
        
    def __del__(self):
        self.video.release()

    def getframe(self,inG=False):
        _,frame=self.video.read()
        if inG:
            frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        return frame
#on last edit
def face_cut(image,coord):

    for (x,y,w,h) in coord:

        off=30
        face=image[y-off:y+off+h+off,x-off:x+w+off]
        face=imutils.resize(face,width=250,inter=cv2.INTER_CUBIC)
        return face

def normalize(images):
    img_norm=[]
    for img in images:
        clr=len(img.shape)==3
        if clr:
            img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img_norm.append(cv2.equalizeHist(img))
    return img_norm

def resize(images,size=(50,50)):
    img_norm=[]
    for image in images:
        if image.shape<size:
            image_norm=cv2.resize(image,size)
        else:
            image_norm=cv2.resize(image,size)
        img_norm.append(image_norm)
    return img_norm

    
if __name__=="__main__":

    cv2.destroyAllWindows()















