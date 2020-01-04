# coding: utf-8

# ================================
# PCA人脸识别主程序
# SEU-PR // R.YY & Z.HF & Z.X
# ================================

import face_detection as FD
import face_recognition as FR
import cv2
import numpy as np
import os
import pickle
import numpy.linalg as la
import matplotlib.pyplot as plt

# 降维后的数据集
fp = open('./model/lower_d_AA100.pkl', 'rb')
lower_dimension_raw_mat = pickle.load(fp)
fp.close()
# 降维矩阵
fp = open('./model/t_base_AA100.pkl', 'rb')
t_base = pickle.load(fp)
fp.close()
# 平均脸
fp = open('./model/mean_face_AA100.pkl', 'rb')
mean_face = pickle.load(fp)
fp.close()

'''
plt.imshow(np.rot90(np.reshape(mean_face, (100, 100)), -1), cmap='gray')
plt.show()

recon = (lower_dimension_raw_mat * t_base.T) + mean_face
recon = np.abs(recon)
plt.imshow(np.rot90(np.reshape(recon[20], (100, 100)), -1), cmap='gray')
plt.show()
'''

def flatten_mat_by_col(mat):
  res = []
  for i in range(mat.shape[1]):
    for j in range(mat.shape[0]):
      res.append(mat[j][i])
  return res


# 脸图预处理
IMAGE_PATH = './face/tests/raw/zhy.png'
whole = cv2.imread(IMAGE_PATH)  # 读彩色
# detected = FD.detect_face(whole)
# if len(detected) != 1:
#   raise ValueError("Too many face.")
detected = [cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)]
face = cv2.equalizeHist(cv2.resize(detected[0], (100, 100)))
origin_face = face.copy()
plt.subplot(141)
plt.title('Origin')
plt.imshow(origin_face, cmap='gray')

# 减去平均脸，然后降维
# face = flatten_mat_by_col(face) - mean_face
face = face.reshape((10000)) - mean_face
face_lowerd = face * t_base

plt.subplot(144)
plt.imshow(np.reshape(face, (100, 100)), cmap='gray')

# 根据距离分类
distance = []
for i in range(47):
  # distance.append(la.norm(np.array(face_lowerd)[0] - np.array(lower_dimension_raw_mat[i])[0]))
  distance.append(FR.distance_norm_L2(face_lowerd, lower_dimension_raw_mat[i]))
arg_sort = np.argsort(np.abs(distance))

clazz = arg_sort[0]
face_file_list = os.listdir('./face/square_gray')
target = cv2.imread(os.path.join('./face/square_gray', face_file_list[clazz]), cv2.IMREAD_GRAYSCALE)

plt.subplot(142)
plt.title('1st like')
plt.imshow(target, cmap='gray')

clazz2 = arg_sort[1]
target = cv2.imread(os.path.join('./face/square_gray', face_file_list[clazz2]), cv2.IMREAD_GRAYSCALE)

plt.subplot(143)
plt.title('2nd like')
plt.imshow(target, cmap='gray')

print("Delta distance = ", distance[clazz2] - distance[clazz])

plt.show()
