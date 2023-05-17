import numpy as np
import cv2

# 顔認識分類器
faceCascade = cv2.CascadeClassifier("D:/OpenCV/opencv/"
                                     "sources/data/haarcascades/haarcascade_frontalface_default.xml")

# 目認識分類器
eyeCascade = cv2.CascadeClassifier("D:/OpenCV/opencv/"
                                     "sources/data/haarcascades/haarcascade_eye.xml")

# カメラをオンにする。
cap = cv2.VideoCapture(0)
ok = True
result = []
while ok:
    # カメラ内の画像を読み取る　okは読み取りが成功したかどうかの判定パラメータ。
    ok, img = cap.read()
    # グレースケール画像に変換する
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 顔検出
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(32, 32)
    )

    # 顔検出に基づいて目を検出する
    for (x, y, w, h) in faces:
        fac_gray = gray[y: (y+h), x: (x+w)]
        result = []
        eyes = eyeCascade.detectMultiScale(fac_gray, 1.3, 2)

        # 目の座標の変換、相対位置から絶対位置への変更
        for (ex, ey, ew, eh) in eyes:
            result.append((x+ex, y+ey, ew, eh))

    # 四角形を描く
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    for (ex, ey, ew, eh) in result:
        cv2.rectangle(img, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

    cv2.imshow('video', img)

    k = cv2.waitKey(1)
    if k == 27:    # press 'ESC' to quit
        break

cap.release()
cv2.destroyAllWindows()
