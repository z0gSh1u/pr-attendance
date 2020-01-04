# coding: utf-8

# ================================
# 测试在测试集上的工作能力
# SEU-PR // R.YY & Z.HF & Z.X
# ================================


import cv2
from FaceNet import FaceNetWrap, convert_gray_to_bgr_use_img, IGNORE_THRESHOLD
import os

helper: FaceNetWrap = None
result = []
ok = 0
filelist = os.listdir('./dataset/test2')
images = []
labels = []


def init():
  global helper
  helper = FaceNetWrap()


init()


# 本地文件方式
def capture_file(img, corrname):
  global detected, result, ok
  # 开始识别过程
  embedding = helper.get_embedding(convert_gray_to_bgr_use_img(img))
  name, mindis, origin_face = helper.get_best_fit(embedding)
  if mindis >= IGNORE_THRESHOLD:
    pass
  else:
    name = name.strip()
    corrname = corrname.strip()
    print("Need=", corrname, ", Got=", name)
    if name == corrname:
      ok += 1


for file in filelist:
  name = file.split('-')[1]
  labels.append(name)
  images.append(cv2.imread(os.path.join('./dataset/test2', file), cv2.IMREAD_GRAYSCALE))
for i in range(len(images)):
  capture_file(images[i], labels[i])

print("OK=", ok)
print("Tot=", len(images))
print("Rate=", ok / float(len(images)) * 100, '%')
