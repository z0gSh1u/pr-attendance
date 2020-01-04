# coding: utf-8

# ================================
# FaceNet人脸识别
# SEU-PR // R.YY & Z.HF & Z.X
# ================================

from keras_facenet import FaceNet
import cv2
import os
import numpy as np

IGNORE_THRESHOLD = 1.  # 距离阈值，若最小距离大于该阈值，舍去该脸


class FaceNetWrap:
  def __init__(self, callback=None):
    self.file_list = os.listdir('./dataset/train')
    self.faces = []
    self.names = []  # faces - names相同下标位置一一对应
    for file in self.file_list:
      self.faces.append(convert_gray_to_bgr_use_path(os.path.join('./dataset/train', file)))
      self.names.append(file.split('.')[0].split('-')[1])
    self.embedder = FaceNet()
    self.embeddings = self.embedder.embeddings(self.faces)
    if callback:
      callback()

  def get_embedding(self, img):
    embeddings_test = self.embedder.embeddings([img])
    return embeddings_test[0]

  # 返回匹配人名和距离、照片
  def get_best_fit(self, embedding):
    dis = [get_L2_norm_squared(em, embedding) for em in self.embeddings]
    idx = np.argmin(dis)
    img = cv2.imread(os.path.join('./dataset/train', self.file_list[idx]), cv2.IMREAD_GRAYSCALE)
    name = self.names[idx]
    return name, dis[idx], img


# 灰度图转三通道图，FaceNet要求
def convert_gray_to_bgr_use_path(path):
  img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
  img = cv2.resize(img, (100, 100))
  img = cv2.equalizeHist(img)
  img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
  return img


def convert_gray_to_bgr_use_img(img):
  img = cv2.resize(img, (100, 100))
  img = cv2.equalizeHist(img)
  img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
  return img


# 计算L2范数之平方作为距离值
def get_L2_norm_squared(a, b):
  sum = 0
  for i in range(len(a)):
    sum += (a[i] - b[i]) * (a[i] - b[i])
  return sum
