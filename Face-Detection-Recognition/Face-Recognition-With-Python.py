# In this program, we’ll look at a surprisingly simple way to get started
# with face recognition using Python and the open source library OpenCV
# This program has been written with the help of RalPython Tutorials

import cv2 #originally written in C/C++
import sys

#To get around this, OpenCV uses cascades. What’s a cascade? The best answer can be found in the
#dictionary: “a waterfall or series of waterfalls.”
#Like a series of waterfalls, the OpenCV cascade breaks the problem of detecting faces into multiple stages. For each block, it does a very rough and quick test. If that passes, it does a slightly more detailed test,
#and so on. The algorithm may have 30 to 50 of these stages or cascades, and it will only detect a face if all stages pass.

imagePath = sys.argv[1]
cascPath = "haarcascade_frontalface_default.xml"

#Create the haar cascade
#Remember, the cascade is just an XML file that contains the data to detect faces.
faceCascade = cv2.CascadeClassifier(cascPath)

#Read the image
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

#Detect faces in the image

faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5,minSize=(30,30), flags=cv2.CASCADE_SCALE_IMAGE)

print ("Found {0} faces!".format(len(faces)))

#Draw a rectangle

for (x,y,w,h) in faces:
    cv2.rectangle(image,(x,y), (x+w,y+h), (0,255,0),2)

cv2.imwrite("detected_image.jpg", image)
