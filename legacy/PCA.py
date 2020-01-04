# coding: utf-8

# ================================
# 利用PCA主成分分析进行人脸识别
# SEU-PR // R.YY & Z.HF & Z.X
# ================================

# 人脸数据库统一使用100X100灰度人脸

import numpy as np

'''
  对数据矩阵进行主成分分析
'''


def PCA(data_mat, k):
  mean_vals = np.mean(data_mat, axis=0)
  mean_removed = data_mat - mean_vals
  cov_mat = np.cov(mean_removed, rowvar=1)
  eig_vals, eig_vects = np.linalg.eig(np.mat(cov_mat))
  eig_val_idx = np.argsort(eig_vals)
  eig_val_idx = eig_val_idx[:-(k + 1):-1]
  re_eig_vects = eig_vects[:, eig_val_idx]
  low_dim_data = re_eig_vects.T * mean_removed
  return low_dim_data, mean_vals, re_eig_vects


# compute the distance between vectors using euclidean distance
def compute_distance(vector1, vector2):
  return np.linalg.norm(np.array(vector1)[0] - np.array(vector2)[0])


# compute the distance between vectors using cosine distance
def compute_distance_(vector1, vector2):
  return np.dot(np.array(vector1)[0], np.array(vector2)[0]) / (
    np.linalg.norm(np.array(vector1)[0]) * (np.linalg.norm(np.array(vector2)[0])))


def test_img(img, mean_vals, low_dim_data):
  mean_removed = img - mean_vals
  return mean_removed * low_dim_data.T
