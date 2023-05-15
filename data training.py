import os
import cv2
from PIL import Image
import numpy as np


def getlable(path):
    facesamples = []  # 储存人脸数据(该数据为二位数组)
    ids = []  # 储存星门数据
    imagepaths = [os.path.join(path, f) for f in os.listdir(path)]  # 储存图片信息
    face_detector = cv2.CascadeClassifier("D:/OpenCV/opencv/"
                                     "sources/data/haarcascades_cuda/haarcascade_frontalface_default.xml")  # 加载分类器
    print('数据排列：', imagepaths)  # 打印数组imagepaths
    for imagePath in imagepaths:  # 遍历列表中的图片
        pil_img = Image.open(imagePath).convert('L')
        # 打开图片，灰度化，PIL的两种不同模式：
        # (1)1(黑白，有像素的地方为1，无像素的地方为0)
        # (2)L(灰度图像，把每个像素点变成0~255的数值，颜色越深值越大)
        img_numpy = np.array(pil_img, 'uint8')  # 将图像转化为数组
        faces = face_detector.detectMultiScale(img_numpy)  # 获取人脸特征
        id = int(os.path.split(imagePath)[1].split('.')[0])  # 获取每张图片的id和姓名
        for x, y, w, h in faces:  # 预防无面容照片
            ids.append(id)
            facesamples.append(img_numpy[y:y + h, x:x + w])
        # 打印脸部特征和id
        print('id:', id)
    print('fs:', facesamples)
    return facesamples, ids


if __name__ == '__main__':
    path = 'D:/BaiduNetdiskDownload/python/opencv/pythonProject/face1/data'  # 图片路径
    faces, ids = getlable(path)  # 获取图像数组和id标签数组和姓名
    recognizer = cv2.face.LBPHFaceRecognizer_create()  # 获取训练对象
    recognizer.train(faces, np.array(ids))
    recognizer.write('trainer/trainer.yml')  # 保存生成的人脸特征数据文件

