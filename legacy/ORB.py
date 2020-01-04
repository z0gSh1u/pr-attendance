# coding: utf-8

# ================================
# 利用ORB特征进行人脸识别
# SEU-PR // R.YY & Z.HF & Z.X
# ================================


import cv2
import numpy as np

img1 = cv2.imread("face/tests/leak/yw.png", cv2.IMREAD_GRAYSCALE)
img1 = cv2.equalizeHist(img1)
img2 = cv2.imread("more/5-YinWei-1.jpg", cv2.IMREAD_GRAYSCALE)
img2 = cv2.equalizeHist(img2)

# ORB Detector
orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)
nfeature = 15
matching_result = cv2.drawMatches(img1, kp1, img2, kp2, matches[:nfeature], None, flags=2)
cv2.imshow("Matching result", matching_result)
cv2.waitKey(0)
cv2.destroyAllWindows()