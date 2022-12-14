from itertools import count
import cv2
import sqlite3



#cam=cv2.VideoCapture(0)
image_path='15.jpg'

face_detector=cv2.CascadeClassifier('xml/haarcascade_frontalface_alt.xml')


img=cv2.imread(image_path)

img_gray=cv2.cvtColor(src=img,code=cv2.COLOR_BGR2GRAY)

while True:
    count=0
    img_gray=cv2.cvtColor(src=img,code=cv2.COLOR_BGR2GRAY)
    faces=face_detector.detectMultiScale(img_gray,1.3,1)

    for(x,y,w,h) in faces:
        x-=30
        y-=30
        w+=60
        h+=60
        img_face=cv2.resize(img[y+3:y+h-3,x+3:x+w-3],(70,70))
        cv2.imwrite('img/people_{}.jpg'.format(count),img_face)
        count +=1
        #cv2.imwrite("dataSet/User."+id +'.'+ str(count) + ".jpg", img_gray[y:y+h,x:x+w])
        cv2.imshow('Face recongition',img)
    for(x,y,w,h) in faces:
        img_face=cv2.resize(img[y+3:y+h-3,x+3:x+w-3],(70,70)) 
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        count +=1
        #cv2.imwrite("dataSet/User."+id +'.'+ str(count) + ".jpg", img_gray[y:y+h,x:x+w])
        cv2.imshow('Face recongition',img)

   
    if cv2.waitKey(delay=0): break
#cam.release()

cv2.destroyAllWindows()









