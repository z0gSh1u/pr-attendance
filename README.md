# 人脸识别考勤系统

本项目为东南大学《模式识别》课程大作业。利用Haar分类器完成人脸检测、分割；利用FaceNet网络完成人脸识别。

<img src="https://s2.ax1x.com/2020/01/05/l0jcAP.png" alt="demo" />

## 开始

1. 以下是本项目的依赖库：

   - opencv-python

   - numpy

   - keras-facenet（见https://pypi.org/project/keras-facenet/）

   - Keras

   - TensorFlow

   
	其中，keras-facenet需要下载预训练模型置于`~/.keras-facenet`目录下，如果你获得的版本在`model/`目录下没有带该模型，请自行到该库的GitHub仓库页下载，或在第一次调用该库时也会自动下载。
	
2. 使用`face_manager.py`可以进行人脸的录入，注意录入姓名时，之间不要用空格分隔。

3. 使用`main.py`可以进行人脸考勤主操作。

## 目录结构

```
├─dataset
│  ├─classroom  测试用多人大图（涉及隐私，不上传）
│  ├─test  测试集1（涉及隐私，不上传）
│  ├─test2  测试集2（涉及隐私，不上传）
│  └─train  训练集（涉及隐私，不上传）
├─legacy  尝试过的其他方法（PCA、SVM、SIFT、KNN、LBPF、FISHER）
├─lib  库安装包
└─model  预训练模型
  └─.keras-facenet  预训练的FaceNet（过大，不上传）
    └─20180402-114759
face_detection.py  Haar人脸检测
face_manager.py  人脸数据管理
FaceNet.py  FaceNet包装
main.py  考勤系统主程序
test.py  准确率测试程序
```

## 效果

使用41张测试集进行测试，最终准确率为**90.24%**。

## 读物

- FaceNet: A Unified Embedding for Face Recognition and Clustering

