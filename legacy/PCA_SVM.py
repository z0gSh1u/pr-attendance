# coding: utf-8

# ================================
# 利用PCA+SVM进行人脸识别
# SEU-PR // R.YY & Z.HF & Z.X
# ================================


from time import time
import logging
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import cv2

# load X, Y
file_list = os.listdir('./face/mo')
X = []
Y = []
for file in file_list:
  img = cv2.imread(os.path.join('./face/mo2', file), cv2.IMREAD_GRAYSCALE)
  img = cv2.resize(img, (100, 100))
  img = cv2.equalizeHist(img)
  img = np.array(img).reshape((10000))
  X.append(img)
  Y.append([str(file).split('-')[0]])
Y = np.array(Y)
Y = Y.reshape((len(file_list)))
X = np.array(X)

# PCA
n_components = 100
h = 100
w = 100
pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True) \
  .fit(X)
eigenfaces = pca.components_.reshape((n_components, h, w))
X_pca = pca.transform(X)

# SVM
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
clf = GridSearchCV(
  SVC(kernel='rbf', class_weight='balanced'), param_grid
)
clf = clf.fit(X_pca, Y)
print("Best estimator found by grid search:")
print(clf.best_estimator_)

xx = cv2.imread('./face/tests/raw/zzy.png', cv2.IMREAD_GRAYSCALE)
xx = cv2.equalizeHist(cv2.resize(xx, (100, 100)))
xx = np.array(xx).reshape(10000)
x_test = [xx]
x_test = np.array(x_test)
X_pca = pca.transform(x_test)

y_pred = clf.predict(np.array(X_pca))
print(y_pred)

print(classification_report(Y, y_pred, target_names=X))
