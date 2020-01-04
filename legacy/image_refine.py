# coding: utf-8

# ================================
# 样本图直方图均衡
# SEU-PR // R.YY & Z.HF & Z.X
# ================================

import cv2
import os

DATASET_PATH = './face/square_gray'

file_list = os.listdir(DATASET_PATH)

i = 0
for file in file_list:
  file_path = os.path.join(DATASET_PATH, file)
  img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
  refine = cv2.equalizeHist(img)
  cv2.imwrite(file_path, refine)
  i += 1

print("Finish.")