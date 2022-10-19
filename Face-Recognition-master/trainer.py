#Copyright Anirban Kar (anirbankar21@gmail.com)
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

from unicodedata import name
import cv2 
import os
import numpy as np
from PIL import Image 
import glob

path = os.path.dirname(os.path.abspath(__file__))
#print(dir(cv2.face))
recognizer = cv2.face.LBPHFaceRecognizer_create()
cascadePath = path+r"/Classifiers/face.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
dataPath = path+r'/dataSet'

def get_images_and_labels(datapath):
    image_paths=[]
    for i in range(1,5):
        name=str(i)
        image_paths0 = ["dataSet/face-" +name +'.'+ str(i) for i in range(1,102) ]
        image_paths1=[s+".jpg" for s in image_paths0]
        image_paths.extend(image_paths1)


    print(len(image_paths))
        


    print(image_paths)
    #image_paths = [f for f in glob.glob(dataPath+'*.jpg')]
    
    
    # image_paths=[
    # "dataSet/face-1.1.jpg",
    # "dataSet/face-1.2.jpg",
    # "dataSet/face-1.3.jpg",
    # "dataSet/face-1.4.jpg",
    # "dataSet/face-1.5.jpg",
    # "dataSet/face-1.6.jpg",
    # "dataSet/face-1.7.jpg",
    # "dataSet/face-1.8.jpg",
    # "dataSet/face-1.9.jpg",
    # "dataSet/face-1.10.jpg",
    # "dataSet/face-1.11.jpg",
    # "dataSet/face-2.1.jpg",
    # "dataSet/face-2.2.jpg",
    # "dataSet/face-2.3.jpg",
    # "dataSet/face-2.4.jpg",
    # "dataSet/face-2.5.jpg",
    # "dataSet/face-2.6.jpg",
    # "dataSet/face-2.7.jpg",
    # "dataSet/face-2.8.jpg",
    # "dataSet/face-2.9.jpg",
    # "dataSet/face-2.10.jpg",
    # "dataSet/face-2.11.jpg",
    #  "dataSet/face-2.1.jpg",
    # "dataSet/face-2.2.jpg",
    # "dataSet/face-2.3.jpg",
    # "dataSet/face-2.4.jpg",
    # "dataSet/face-2.5.jpg",
    # "dataSet/face-2.6.jpg",
    # "dataSet/face-2.7.jpg",
    # "dataSet/face-2.8.jpg",
    # "dataSet/face-2.9.jpg",
    # "dataSet/face-2.10.jpg",
    # "dataSet/face-2.11.jpg",
    # "dataSet/face-3.1.jpg",
    # "dataSet/face-3.2.jpg",
    # "dataSet/face-3.3.jpg",
    # "dataSet/face-3.4.jpg",
    # "dataSet/face-3.5.jpg",
    # "dataSet/face-3.6.jpg",
    # "dataSet/face-3.7.jpg",
    # "dataSet/face-3.8.jpg",
    # "dataSet/face-3.9.jpg",
    # "dataSet/face-3.10.jpg",
    # "dataSet/face-3.11.jpg",
    # "dataSet/face-4.1.jpg",
    # "dataSet/face-4.2.jpg",
    # "dataSet/face-4.3.jpg",
    # "dataSet/face-4.4.jpg",
    # "dataSet/face-4.5.jpg",
    # "dataSet/face-4.6.jpg",
    # "dataSet/face-4.7.jpg",
    # "dataSet/face-4.8.jpg",
    # "dataSet/face-4.9.jpg",
    # "dataSet/face-4.10.jpg",
    # "dataSet/face-4.11.jpg",
    # "dataSet/face-5.1.jpg",
    # "dataSet/face-5.2.jpg",
    # "dataSet/face-5.3.jpg",
    # "dataSet/face-5.4.jpg",
    # "dataSet/face-5.5.jpg",
    # "dataSet/face-5.6.jpg",
    # "dataSet/face-5.7.jpg",
    # "dataSet/face-5.8.jpg",
    # "dataSet/face-5.9.jpg",
    # "dataSet/face-5.10.jpg",
    # "dataSet/face-5.11.jpg",
    # ]
     
     # images will contains face images
    images = []
     # labels will contains the label that is assigned to the image
    labels = []
    for image_path in image_paths:
         # Read the image and convert to grayscale
        image_pil = Image.open(image_path)
        image_pil.convert('L')
         # Convert the image format into numpy array
        image = np.array(image_pil, 'uint8')
         # Get the label of the image
        nbr = int(os.path.split(image_path)[1].split(".")[0].replace("face-", ""))
         #nbr=int(''.join(str(ord(c)) for c in nbr))
        print(nbr)
         # Detect the face in the image
        faces = faceCascade.detectMultiScale(image)
         # If face is detected, append the face to images and the label to labels
        for (x, y, w, h) in faces:
             images.append(image[y: y + h, x: x + w])
             labels.append(nbr)
             cv2.imshow("Adding faces to traning set...", image[y: y + h, x: x + w])
             cv2.waitKey(10)
     # return the images list and labels list
    return images, labels


images, labels = get_images_and_labels(dataPath)
cv2.imshow('test',images[0])
cv2.waitKey(1)

recognizer.train(images, np.array(labels))
recognizer.save(path+r'/trainer/trainer.yml')
cv2.destroyAllWindows()
