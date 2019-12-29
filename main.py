# coding: utf-8

# ================================
# 人脸识别主程序
# SEU-PR // R.YY & Z.HF & Z.X
# ================================

import face_recognition as FR
import cv2
import time
import os
import pickle

import matplotlib.pyplot as plt

N_FEATURE = 100

lower_dimension_raw_mat = None
transform_base = None


# 把47个样本拉伸成47X(100X100)的数据矩阵，即47行，10000列，图像按列展开，列间首尾相接
def trian():
  # 按列展开一个矩阵
  def flatten_mat_by_col(mat):
    res = []
    for i in range(mat.shape[1]):
      for j in range(mat.shape[0]):
        res.append(mat[j][i])
    return res

  global lower_dimension_raw_mat, transform_base
  # 组织初始数据矩阵 47X10000
  face_file_list = os.listdir('./face/square_gray')
  raw_mat = []
  for face_file in face_file_list:
    image = cv2.imread(os.path.join('./face/square_gray', face_file), cv2.IMREAD_GRAYSCALE)
    raw_mat.append(flatten_mat_by_col(image))
  # 训练之
  start_time = time.time()
  print("Start training...")
  lower_dimension_raw_mat, transform_base = FR.PCA(raw_mat, N_FEATURE)
  end_time = time.time()
  print("End training... Used time = " + str((end_time - start_time) / (1000 * 60)) + " min.")
  print("Saving...")

  f1 = open('lower_d.pkl', 'wb')
  pickle.dump(lower_dimension_raw_mat, f1)
  f1.close()
  f1 = open('t_base.pkl', 'wb')
  pickle.dump(transform_base, f1)
  f1.close()

  print("Dump OK...")

trian()
