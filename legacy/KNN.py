# coding: utf-8

# ================================
# 利用KNN进行人脸识别
# SEU-PR // R.YY & Z.HF & Z.X
# ================================


import cv2
from sklearn.neighbors import KNeighborsClassifier
import os
import numpy as np
import matplotlib.pyplot as plt

filelist = os.listdir('./face/mo')

X = []
Y = []

corr = {}

for file in filelist:
  img = cv2.imread(os.path.join('./face/mo', file), cv2.IMREAD_GRAYSCALE)
  img = cv2.resize(img, (100, 100))
  img = cv2.equalizeHist(img)
  img_flat = np.reshape(img, (10000))
  i = int(str(file).split('-')[0])
  Y.append(i)
  corr[i] = str(file)
  X.append(img_flat)

X = np.array(X)
Y = np.array(Y).reshape(-1, 1).ravel()

model = KNeighborsClassifier(weights='distance')
model.fit(X, Y)

test_img = cv2.imread('./face/tests/raw/1577953826.png', cv2.IMREAD_GRAYSCALE)
test_img = cv2.equalizeHist(cv2.resize(test_img, (100, 100)))
bak = test_img.copy()
test_img = np.reshape(test_img, (10000))
test_img = np.array(test_img).reshape(1, -1)

response = model.predict(test_img)

print(response)
plt.subplot(121)
plt.imshow(bak, cmap='gray')
plt.subplot(122)
plt.imshow(cv2.imread(os.path.join('./face/mo', corr[response[0]]), cv2.IMREAD_GRAYSCALE), cmap='gray')
plt.show()
