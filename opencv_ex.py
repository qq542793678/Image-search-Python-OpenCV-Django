# coding:utf-8

import cv2
import numpy as np

img = cv2.imread('gg.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT()
# 找到图像上的关键点
kp = sift.detect(gray, None)
# 绘制关键点
img = cv2.drawKeypoints(gray, kp)

kp, des = sift.compute(gray, kp)
# kp关键点列表.des是一个Numpy数组
print 'kp:', kp
print 'des', des

cv2.imwrite('sift_keypoints.jpg', img)


