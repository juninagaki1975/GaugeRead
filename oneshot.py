#!/usr/bin/env python3
#
# Jun 2024.6.11
#

import cv2
from datetime import datetime as dt

deviceid = 0
cap = cv2.VideoCapture(deviceid)

timestamp = dt.now()
fname =  timestamp.strftime('%Y%m%d%H%M%S') + '.jpg'
ret, frame = cap.read()

height = frame.shape[0]
width = frame.shape[1]
center = (int(width/2),int(height/2))

theta = -90.0
scale = 1.0
rot = cv2.getRotationMatrix2D(center, theta, scale)
image2 = cv2.warpAffine(frame, rot, (width, height))

#cv2.imwrite("test.jpg", frame)
cv2.imwrite(fname,image2)
