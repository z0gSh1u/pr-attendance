# coding: utf-8

# ================================
# 利用PCA主成分分析进行人脸识别 - 训练器
# SEU-PR // R.YY & Z.HF & Z.X
# ================================


import face_recognition as FR
import cv2 as cv
from face_recognition import *
import time
import os
import pickle
import numpy as np

# 保留多少特征
N_FEATURE = 10

# 1. use num 1- 9 image of each person to train
data = []
filelist = os.listdir('./face/square_gray')

for file in filelist:
  print(file)
  img = cv.imread(os.path.join('./face/square_gray', file), cv.IMREAD_GRAYSCALE)
  img = cv.resize(img, (100, 100))
  width, height = img.shape
  img = img.reshape((img.shape[0] * img.shape[1]))
  data.append(img)

low_dim_data, mean_vals, re_eig_vects = PCA(data, N_FEATURE)

fp1 = open('./ldd.pkl', 'wb')
fp2 = open('./mv.pkl', 'wb')
fp3 = open('./rev.pkl', 'wb')
pickle.dump(low_dim_data, fp1)
pickle.dump(mean_vals, fp2)
pickle.dump(re_eig_vects, fp3)
fp1.close()
fp2.close()
fp3.close()
