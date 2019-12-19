# coding: utf-8

# 利用PCA主成分分析进行人脸识别

import numpy as np

def PCA(data, k):
  data = np.asarray(data, dtype=np.float)
  row, col = data.shape
  data_mean = np.mean(data, 0)

