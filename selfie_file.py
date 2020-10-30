import cv2
import numpy as np
import dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("../shape_predictor_68_face_landmarks.dat")
(mStart,mEnd)=(48,67)
smile_const=5
counter = 0
selfie_no = 0
def rect_to_bb(rect):
    x=rect.left()
    y=rect.top()
    w=rect.right()- x
    h=rect.bottom()- y
    return (x,y,w,h)
def shape_to_np(shape,dtype='int'):
    coords=np.zeros((68,2),dtype=dtype)
    for i in range(0,68):
        coords[i]=(shape.part(i).x,shape.part(i).y)
    return coords
def smile(shape):
    left=shape[48]
    right=shape[54]
    mid=(shape[51]+shape[62]+shape[66]+shape[57])/4
    dist=np.abs(np.cross(right-left,left-mid)/np.linalg.norm(right-left))
    return dist
cam=cv2.VideoCapture(0)
while(cam.isOpened()):
    ret,image=cam.read()
    image=cv2.flip(image,1)
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    rects=detector(gray,2)
    for i in range(0,len(rects)):
        (x,y,w,h)=rect_to_bb(rects[i])
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(image,"Face #{}".format(i+1),(x-10,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,(233,234,255),3)
        shape = predictor(gray,rects[i])
        shape=shape_to_np(shape)
        mouth= shape[mStart:]
        for (x,y) in mouth:
            cv2.circle(image,(x,y),1,(233,212,255),-1)
        smile_param = smile(shape)
        cv2.putText(image,"sp:{:2f}".format(smile_param),(300,30),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),3)
        if smile_param>smile_const:
            cv2.putText(image,"keep smiling !! smile is detected yayyy",(300,60),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,225,0),2)
            counter +=1
            if counter >=2:
                selfie_no +=1
                ret,frame = cam.read()
                img_name = "smart_selfie_{}.png".format(selfie_no)
                cv2.imwrite(img_name,frame)
                print("{} pic taken!! congratulations buddy".format(img_name))
                counter = 0
                
            else:
                counter = 0
                
    cv2.imshow('live_face',image)
    key=cv2.waitKey(1)
    if key==27:
        break
cam.release()
cv2.destroyAllWindows()

