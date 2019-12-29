# coding: utf-8

# ================================
# 利用PCA主成分分析进行人脸识别
# SEU-PR // R.YY & Z.HF & Z.X
# ================================

# 人脸数据库统一使用100X100灰度人脸
# 保留15维特征？

import numpy as np
import matplotlib.pyplot as plt

'''
  对数据矩阵进行主成分分析
'''


def PCA(data_mat, target_n_feature):
  mean_val = np.mean(data_mat, axis=0)
  # 平均脸，有需要则释放该注释
  # mean_face = np.reshape(mean_val, (100, 100))
  mean_removed = data_mat - mean_val  # STEP1 - 减去均值，去中心化
  cov_mat = np.cov(mean_removed, rowvar=0)  # STEP2 - 计算协方差矩阵
  eig_vals, eig_vecs = np.linalg.eig(np.mat(cov_mat))  # STEP3 - 计算特征值和特征向量
  eig_val_index = np.argsort(eig_vals)
  eig_val_index = eig_val_index[:-(target_n_feature + 1):-1]  # 降维，裁剪特征
  reorganized_eig_vecs = eig_vecs[:, eig_val_index]  # 新的基
  lower_dimension_mat = mean_removed * reorganized_eig_vecs  # STEP4 - 空间转换
  # reconstructed_mat = (lower_dimension_mat * reorganized_eig_vecs.T) + mean_val  # STEP5 - 重构
  return lower_dimension_mat, reorganized_eig_vecs  # 返回降维后的脸和降维用的基
