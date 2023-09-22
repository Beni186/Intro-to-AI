import os
from turtle import Turtle
import cv2
import matplotlib.pyplot as plt
import numpy as np
import utils
from os import walk
from os.path import join
from datetime import datetime


def crop(x1, y1, x2, y2, x3, y3, x4, y4, img) :
    """
    This function ouput the specified area (parking space image) of the input frame according to the input of four xy coordinates.
    
      Parameters:
        (x1, y1, x2, y2, x3, y3, x4, y4, frame)
        
        (x1, y1) is the lower left corner of the specified area
        (x2, y2) is the lower right corner of the specified area
        (x3, y3) is the upper left corner of the specified area
        (x4, y4) is the upper right corner of the specified area
        frame is the frame you want to get it's parking space image
        
      Returns:
        parking_space_image (image size = 360 x 160)
      
      Usage:
        parking_space_image = crop(x1, y1, x2, y2, x3, y3, x4, y4, img)
    """
    left_front = (x1, y1)
    right_front = (x2, y2)
    left_bottom = (x3, y3)
    right_bottom = (x4, y4)
    src_pts = np.array([left_front, right_front, left_bottom, right_bottom]).astype(np.float32)
    dst_pts = np.array([[0, 0], [0, 160], [360, 0], [360, 160]]).astype(np.float32)
    projective_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    croped = cv2.warpPerspective(img, projective_matrix, (360,160))
    return croped

def draw_box(img, x1, y1, x2, y2, x3, y3, x4, y4):
    color = (0, 255, 0) 
    thickness = 2 
    cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    cv2.line(img, (x2, y2), (x4, y4), color, thickness)
    cv2.line(img, (x3, y3), (x4, y4), color, thickness)
    cv2.line(img, (x1, y1), (x3, y3), color, thickness)

def detect(dataPath, clf):
    """
    Please read detectData.txt to understand the format. 
    Use cv2.VideoCapture() to load the video.gif.
    Use crop() to crop each frame (frame size = 1280 x 800) of video to get parking space images. (image size = 360 x 160) 
    Convert each parking space image into 36 x 16 and grayscale.
    Use clf.classify() function to detect car, If the result is True, draw the green box on the image like the example provided on the spec. 
    Then, you have to show the first frame with the bounding boxes in your report.
    Save the predictions as .txt file (Adaboost_pred.txt), the format is the same as GroundTruth.txt. 
    (in order to draw the plot in Yolov5_sample_code.ipynb)
    
      Parameters:
        dataPath: the path of detectData.txt
      Returns:
        No returns.
    """
    # Begin your code (Part 4)
    capture = cv2.VideoCapture(cv2.samples.findFileOrKeep(dataPath+"/video.gif"))

    with open(dataPath+"/detectData.txt",'r') as fh:
        linelist = fh.readlines()
        for i in range(len(linelist)):
            linelist[i] = linelist[i].strip()
            
    result = []
    temp=[]
    tt=[]
    for i in linelist:
        temp= i.split()
        for j in temp:
            j = int(j)
            tt.append(j)
        result.append(tt)
        temp=[]
        tt=[]
    result.pop(0)

    count=0
    while capture.isOpened():
      ret, frame= capture.read()
      if frame is None:
        break
      # frame = cv2.resize(frame, (1200, 800))
      temp=[]
      j=0
      for i in result:
        img = crop(i[0], i[1], i[2], i[3], i[4], i[5], i[6], i[7], frame)
        img = cv2.resize(img, (36,16))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # utils.integralImage(img)
        if j != 0:
          temp.append(" ")
        j+=1
        if clf.classify(img) == 1:
          draw_box(frame, i[0], i[1], i[2], i[3], i[4], i[5], i[6], i[7])
          temp.append("1")
        else:
          temp.append("0")
      if count == 0 :
        f = open("Adaboost_pred.txt", "w")
        # cv2.imwrite("data/detect/first.png", frame)
      else :  
        f = open("Adaboost_pred.txt", "a")
      f.writelines(temp)
      f.write("\n")
      f.close()
      count+=1
      # cv2.imshow('My Image', frame)
      # cv2.waitKey(0)  
    
    #raise NotImplementedError("To be implemented")
    # End your code (Part 4)



