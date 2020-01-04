# coding: utf-8

# ================================
# 人脸样本录入环节
# SEU-PR // R.YY & Z.HF & Z.X
# ================================

import cv2
import PIL.Image
from PIL import ImageTk
from tkinter import *
from tkinter import filedialog, messagebox
import json
import face_detection as FD
from time import time

TARGET_SIZE = (100, 100)
howmany = 0
detected = []

# 摄像头捕获方式
camera = cv2.VideoCapture(0)  # 摄像头
current_status = 'none'
camera_looper = None


def capture_camera():
  global camera, camera_looper, detected, howmany
  if not camera.isOpened():
    camera = cv2.VideoCapture(0)  # 摄像头
  status, frame = camera.read()
  if status:
    cv2.waitKey(10)
    (res, howmany, detected) = FD.detect_face_for_manager(frame)
    label_faceCount.config(text='当前人脸数：' + str(howmany))
    res = cv2.resize(res, (640, int(640. / res.shape[1] * res.shape[0])))
    res = cv2.cvtColor(res, cv2.COLOR_BGR2RGBA)
    res_pil = PIL.Image.fromarray(res)
    res_tk = ImageTk.PhotoImage(image=res_pil)
    display_area.imgtk = res_tk
    display_area.config(image=res_tk, width=res.shape[1], height=res.shape[0])
    camera_looper = root.after(1, capture_camera)  # 反复更新当前帧


# 本地文件方式
def capture_file(path: str):
  global detected, howmany
  if camera.isOpened() and camera_looper:
    root.after_cancel(camera_looper)
    camera.release()
  img = cv2.imread(path)
  (res, howmany, detected) = FD.detect_face_for_manager(img, int(img.shape[0] / 80) + 1)
  label_faceCount.config(text='当前人脸数：' + str(howmany))
  res = cv2.resize(res, (640, int(640. / res.shape[1] * res.shape[0])))
  res = cv2.cvtColor(res, cv2.COLOR_BGR2RGBA)
  res_pil = PIL.Image.fromarray(res)
  res_tk = ImageTk.PhotoImage(image=res_pil)
  display_area.imgtk = res_tk
  display_area.config(image=res_tk, width=res.shape[1], height=res.shape[0])


def ask_open_file():
  path = filedialog.askopenfilename(title='打开图片...', filetypes=[('PNG Image', '*.png'), ('JPG Image', '*.jpg')])
  capture_file(path)


def save_sample():
  global detected, howmany
  if howmany != 1:
    messagebox.showerror('错误', '当前人脸数不为1！')
    return
  face_gray = cv2.resize(detected[0], TARGET_SIZE)
  name = input_label.get()
  fp = open('./dataset/manager.json', 'r')
  json_content = json.load(fp)
  fp.close()
  total = int(json_content['total'])
  total += 1
  json_content['total'] = total
  fp = open('./dataset/manager.json', 'w')
  json.dump(json_content, fp)
  fp.close()
  filename = str(total) + '-' + name
  save_path = './dataset/train/' + filename + '.png'
  cv2.imwrite(save_path, face_gray)
  messagebox.showinfo('录入成功', '录入成功！')


root = Tk()
root.title("人脸样本录入 - [SEU-PR]R.YY, Z.HF & Z.X")
Label(root, {'text': '人脸样本录入 - [SEU-PR]', 'font': '宋体 16'}).pack()
label_faceCount = Label(root, {'text': '当前人脸数：0', 'foreground': 'red'})
label_faceCount.pack()
display_area = Label(root, {'width': 100, 'height': 30, 'text': '请选择一种录入方式开始...'})
display_area.pack()
Button(root, {'text': '从摄像头录入', 'command': capture_camera}).pack({'side': LEFT})
Button(root, {'text': '从本地图片录入', 'command': ask_open_file}).pack({'side': LEFT})
Label(root, {'text': '请确保只有一张需要录入的人脸被框定！', 'foreground': 'red'}).pack({'side': LEFT})
Label(root, {'text': '请输入姓名：'}).pack({'side': LEFT})
input_label = Entry(root)
input_label.pack({'side': LEFT})
Button(root, {'text': '保存', 'command': save_sample}).pack({'side': LEFT})

root.mainloop()
camera.release()
