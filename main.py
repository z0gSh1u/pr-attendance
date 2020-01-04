# coding: utf-8

# ================================
# 考勤系统主程序
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
from FaceNet import FaceNetWrap, convert_gray_to_bgr_use_img, IGNORE_THRESHOLD

detected = []
helper: FaceNetWrap = None
result = []


# 本地文件方式
def capture_file(path: str):
  global detected, result
  result = []
  img = cv2.imread(path)
  if img is None:
    return
  # 先做一次人脸检测，但不渲染结果图
  (_, _, detected, faces) = FD.detect_face_for_manager(img.copy(), rect_width=int(img.shape[0] / 80) + 1)
  # 开始识别过程
  my_mask = []
  member_comes = []
  for face in detected:
    embedding = helper.get_embedding(convert_gray_to_bgr_use_img(face))
    name, mindis, origin_face = helper.get_best_fit(embedding)
    if mindis >= IGNORE_THRESHOLD:
      my_mask.append(False)
    else:
      my_mask.append(True)
      member_comes.append([name, mindis])
  # 画框
  for i in range(len(my_mask)):
    if my_mask[i]:
      (x, y, w, h) = faces[i]
      img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), int(img.shape[0] / 80) + 1)
  # 渲染到GUI
  if img.shape[0] > img.shape[1]:
    target_size = (int(640. / img.shape[0] * img.shape[1]), 640)
  else:
    target_size = (640, int(640. / img.shape[1] * img.shape[0]))
  res = cv2.resize(img, target_size)
  res = cv2.cvtColor(res, cv2.COLOR_BGR2RGBA)
  res_pil = PIL.Image.fromarray(res)
  res_tk = ImageTk.PhotoImage(image=res_pil)
  display_area.imgtk = res_tk
  display_area.config(image=res_tk, width=res.shape[1], height=res.shape[0])
  # 弹窗显示结果
  total_member_count = len(helper.faces)
  come_member_count = len(member_comes)
  attendence_rate = come_member_count / float(total_member_count)  # 出勤率
  top_window = Toplevel()
  top_window.title('考勤结果')
  to_disp = '到场人数：' + str(come_member_count) + '\n总人数：' + str(total_member_count) + '\n出勤率：' + str(
    attendence_rate * 100)[0:6] + '%\n' + '到场学生：\n'
  for (name, _) in member_comes:
    to_disp += '\t'
    to_disp += name
    to_disp += '\t\n'
  Label(top_window, {'text': to_disp, 'font': '宋体 13', 'justify': LEFT}).pack({'side': LEFT})


def ask_open_file():
  path = filedialog.askopenfilename(title='打开图片...', filetypes=[('JPG Image', '*.jpg'), ('PNG Image', '*.png')])
  capture_file(path)


def init():
  def data_loaded_callback():
    display_area.config({'width': 100, 'height': 30, 'text': '初始化完成，请载入图片...', 'font': '黑体 16'})
    btn_start.config({'state': NORMAL})

  global helper
  helper = FaceNetWrap(data_loaded_callback)


root = Tk()
root.title("人脸识别考勤系统主程序 - [SEU-PR]R.YY, Z.HF & Z.X")
Label(root, {'text': '人脸识别考勤系统主程序 - [SEU-PR]', 'font': '宋体 16'}).pack()
display_area = Label(root, {'width': 100, 'height': 30, 'text': '正在初始化数据...', 'font': '黑体 16'})
display_area.pack()
btn_start = Button(root, {'text': '从本地图片录入', 'command': ask_open_file, 'state': DISABLED})
btn_start.pack({'side': LEFT})

root.after(1000, init)
root.mainloop()
