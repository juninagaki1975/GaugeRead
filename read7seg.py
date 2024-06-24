#!/usr/bin/env python3
#
# Jun 2024.6.11
#
# TODO: height width or w, h?

import cv2
import numpy as np
from datetime import datetime as dt
import matplotlib.pyplot as plt

def get_info(frame):

    #height
    h = frame.shape[0]

    #width
    w = frame.shape[1]

    #center
    c = (int(w/2) ,int(h/2))

    return w, h, c


def crop(frame,wmin,wmax,hmin,hmax):

    img = frame[wmin:wmax,hmin:hmax]

    return img


def skew(img,theta):

    a = np.tan(np.deg2rad(theta))
    skew = np.array([
                [1,0,0],
                [a,1,0]
                ],
                    dtype=np.float32)
    h,w,c = get_info(img) 
    img = cv2.warpAffine(img, skew, (h, w))

    return img

def rot(frame,center,theta,scale):

    w,h,c = get_info(frame)
#debug
    rot = cv2.getRotationMatrix2D(center, theta, scale)
    img = cv2.warpAffine(frame, rot, (w,h))

    return img

def contour(frame):

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
#    ret, binary = cv2.threshold(gray,0,255,cv2.THRESH_OTSU)
    ret, binary = cv2.threshold(gray,150,255,cv2.THRESH_BINARY)
#debug
#    plt.imshow(cv2.cvtColor(binary, cv2.COLOR_BGR2RGB))
#    plt.show()

    contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE, \
         cv2.CHAIN_APPROX_NONE)
#    contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE, \
#         cv2.CHAIN_APPROX_SIMPLE)

    img_blank = np.ones_like(frame) * 255
    img_contour_only = cv2.drawContours(img_blank, contours, -1, \
         (0,0,0), 3)
#debug
#    plt.imshow(cv2.cvtColor(img_contour_only, cv2.COLOR_BGR2RGB))
#    plt.show()

    return img_contour_only


if __name__== "__main__" :

### camera set ###

    deviceid = 0
    cap = cv2.VideoCapture(deviceid)
    timestamp = dt.now()
    fname =  timestamp.strftime('%Y%m%d%H%M%S') + '.jpg'
    ret, frame = cap.read()

    w,h,c = get_info(frame)
#debug
    print("height : ",h)
    print("width  : ",w)
    print("center  : ",c)

    img = frame[0:480,0:640]

    theta = 180
    scale = 1.0
    img = rot(img,c,theta,scale)

# T1
    img1 = img[385:455,200:360]
    h,w,c = get_info(img1)
    img1 = rot(img1,c,-4.3,1.0)
    img1 = skew(img1,-4.5)
    img1 = contour(img1)

    plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    plt.show()

    cv2.imwrite("contour.jpg",img1)

