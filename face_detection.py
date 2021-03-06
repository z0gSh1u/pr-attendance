# coding: utf-8

# ================================
# 基于OpenCV提供的Haar分类器的人脸检测
# SEU-PR // R.YY & Z.HF & Z.X
# ================================

# 参数
SCALE_FACTOR = 1.1  # 以多大的比率变换窗口大小
MIN_NEIGHBORS = 3  # 多少次连续出现才确信是脸
MIN_SIZE = (95, 95)  # 最小脸尺寸

import cv2

face_cascade = cv2.CascadeClassifier('./model/haarcascade_frontalface_default.xml')

'''
  检测人脸，返回各个人脸切片图（灰度图）
'''


def detect_face(img):
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  faces = face_cascade.detectMultiScale(gray, scaleFactor=SCALE_FACTOR, minNeighbors=MIN_NEIGHBORS, minSize=MIN_SIZE)
  detected = []
  for (x, y, w, h) in faces:
    cut = gray[y:y + h, x:x + w]
    detected.append(cut)
  return detected


'''
  测试用，返回标框图像和脸个数、脸
'''


def detect_face_for_manager(img, rect_width=3):
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  faces = face_cascade.detectMultiScale(gray, scaleFactor=SCALE_FACTOR, minNeighbors=MIN_NEIGHBORS, minSize=MIN_SIZE)
  howmany = len(faces)
  detected = []
  for (x, y, w, h) in faces:
    img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), rect_width)
    cut = gray[y:y + h, x:x + w]
    detected.append(cut)
  return (img, howmany, detected, faces)
