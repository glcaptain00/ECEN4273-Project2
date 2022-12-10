# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 14:43:33 2022

source:https://medium.com/@basit.javed.awan/resizing-multiple-images-and-saving-them-using-opencv-518f385c28d3

@author: redst
"""

import cv2
import glob
import os

inputfolder = './data/TestingSet'
folderLen = len(inputfolder)
#os.mkdir('./data/TrainingSet/resized')

i = 101

for img in glob.glob(inputfolder + "/*.jpg"):
    image = cv2.imread(img)
    imgResized = cv2.resize(image, (100, 100))
    cv2.imwrite("./data/TestingSet/resized/cat.%04i.jpg" %i, imgResized)
    i +=1
    cv2.imshow('image', imgResized)
    cv2.waitKey(30)
cv2.destroyAllWindows()