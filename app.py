import cv2
import os
from for_model import for_model
from keras.preprocessing.image import img_to_array, load_img
import numpy as np

CASCADE_FILE_PATH = os.environ['CV2_CASCADE_FILE_PATH']


if __name__ == '__main__':
    # 定数定義
    ESC_KEY = 27     # Escキー
    INTERVAL= 33     # 待ち時間
    FRAME_RATE = 30  # fps

    ORG_WINDOW_NAME = "org"
    GAUSSIAN_WINDOW_NAME = "gaussian"

    DEVICE_ID = 0

    # 分類器の指定
    cascade = cv2.CascadeClassifier(CASCADE_FILE_PATH)

    ##貼り付け画像
    anime_file = "smile.png"
    anime_face = cv2.imread(anime_file)

    ##画像を貼り付ける関数
    def anime_face_func(img, rect):
        (x1, y1, x2, y2) = rect
        w = x2 - x1
        h = y2 - y1
        img_face = cv2.resize(anime_face, (w, h))
        img2 = img.copy()
        img2[y1:y2, x1:x2] = img_face
        return img2

    # カメラ映像取得
    cap = cv2.VideoCapture(DEVICE_ID)

    # 初期フレームの読込
    end_flag, c_frame = cap.read()
    height, width, channels = c_frame.shape

    # ウィンドウの準備
    cv2.namedWindow(ORG_WINDOW_NAME)
    cv2.namedWindow(GAUSSIAN_WINDOW_NAME)

    # 変換処理ループ
    while end_flag == True:

        # 画像の取得と顔の検出
        img = c_frame
        # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_list = cascade.detectMultiScale(img, minSize=(100, 100))

        # img_gray  = load_img(img, grayscale=True , target_size=(64, 64))
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.resize(img_gray, (64, 64))
        img_array = img_to_array(img_gray)
        pImg = np.expand_dims(img_array, axis=0) / 255

        # モデルに投げる
        pre = for_model(pImg)
        print(pre)

        if len(pre) == 2 and (pre[0][1] > 0.3 or pre[1][1]) > 0.3:
            for (x, y, w, h) in face_list:
                color = (0, 0, 225)
                pen_w = 3
                # cv2.rectangle(img, (x, y), (x+w, y+h), color, thickness = pen_w)
                img = anime_face_func(img, (x, y, x+w, y+h))
                cv2.putText(img, "Let's smile!", (x,y-30), cv2.FONT_HERSHEY_DUPLEX | cv2.FONT_ITALIC, 2.5, (100,100,200), 4, cv2.LINE_AA)
        elif len(pre) == 1 and pre[0][1] > 0.3:
            for (x, y, w, h) in face_list:
                color = (0, 0, 225)
                pen_w = 3
                # cv2.rectangle(img, (x, y), (x+w, y+h), color, thickness = pen_w)
                img = anime_face_func(img, (x, y, x+w, y+h))
                cv2.putText(c_frame, "Let's smile!", (x,y-30), cv2.FONT_HERSHEY_DUPLEX | cv2.FONT_ITALIC, 2.5, (100,100,200), 4, cv2.LINE_AA)
        # else:
        #     for (x, y, w, h) in face_list:
        #         color = (255, 0, 0)
        #         pen_w = 3
        #         cv2.rectangle(img, (x, y), (x+w, y+h), color, thickness = pen_w)


        # フレーム表示
        cv2.imshow(ORG_WINDOW_NAME, c_frame)
        cv2.imshow(GAUSSIAN_WINDOW_NAME, img)

        # Escキーで終了
        key = cv2.waitKey(INTERVAL)
        if key == ESC_KEY:
            break

        # 次のフレーム読み込み
        end_flag, c_frame = cap.read()

    # 終了処理
    cv2.destroyAllWindows()
    cap.release()
