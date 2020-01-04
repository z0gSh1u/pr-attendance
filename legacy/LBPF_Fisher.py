# coding: utf-8

# ================================
# 利用LBPF特征与Fisher特征进行人脸识别
# SEU-PR // R.YY & Z.HF & Z.X
# ================================


import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

filelist = os.listdir('./face/square_gray')
faces = []
labels = []
for file in filelist:
  img = cv2.imread(os.path.join('./face/square_gray', file), cv2.IMREAD_GRAYSCALE)
  img = cv2.resize(img, (100, 100))
  img = cv2.equalizeHist(img)
  faces.append(img)
  labels.append(int(file.split('.')[0]))

# face_re = cv2.face.LBPHFaceRecognizer_create()
face_re = cv2.face.FisherFaceRecognizer_create()
face_re.train(faces, np.array(labels))
face_re.write(r'./model_lbph.yml')

face_test = cv2.equalizeHist(cv2.resize(cv2.imread('face/mo/1578059815.png', cv2.IMREAD_GRAYSCALE), (100, 100)))
pred = face_re.predict(face_test)

print(pred)
