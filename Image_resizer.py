# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 14:43:33 2022

@author: redst
"""

import cv2
import glob
import os

inputfolder = './data/CatsandDogs/train/mixed'
folderLen = len(inputfolder)
os.mkdir('./data/CatsandDogs/train/resized')

for img in glob.glob(inputfolder + "/*.jpg"):
    image = cv2.imread(img)
    imgResized = cv2.resize(image, (100, 100))
    cv2.imwrite('./data/CatsandDogs/train/resized' + img[folderLen:], imgResized)
    cv2.imshow('image', imgResized)
    cv2.waitKey(30)
cv2.destroyAllWindows()