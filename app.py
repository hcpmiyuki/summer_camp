import cv2
import os
from for_model import for_model
from keras.preprocessing.image import img_to_array, load_img
import numpy as np
import time
# import pygame.mixer
from PIL import Image

CASCADE_FILE_PATH = "haarcascade_frontalface_alt2.xml"


# #効果音を鳴らすための処理
# pygame.mixer.init()
# pygame.mixer.music.load("blackout5.mp3")


if __name__ == '__main__':
    # 効果音出すためのフラグ
    # smile_flag = False

    # 定数定義
    # ESC_KEY = 27     # Escキー
    # INTERVAL= 33     # 待ち時間

    GAUSSIAN_WINDOW_NAME = "gaussian"

    DEVICE_ID = 0

    # 分類器の指定
    cascade = cv2.CascadeClassifier(CASCADE_FILE_PATH)

    ##貼り付け画像
    smile = cv2.imread('smile.png', cv2.IMREAD_UNCHANGED)

    def overlay(background_image, overlay_image, point):
        # OpenCV形式の画像をPIL形式に変換(α値含む)

        # 背景画像
        rgb_background_image = cv2.cvtColor(background_image, cv2.COLOR_BGR2RGB)
        pil_rgb_background_image = Image.fromarray(rgb_background_image)
        pil_rgba_background_image = pil_rgb_background_image.convert('RGBA')
        # オーバーレイ画像
        cv_rgb_overlay_image = cv2.cvtColor(overlay_image, cv2.COLOR_BGRA2RGBA)
        pil_rgb_overlay_image = Image.fromarray(cv_rgb_overlay_image)
        pil_rgba_overlay_image = pil_rgb_overlay_image.convert('RGBA')

        # composite()は同サイズ画像同士が必須のため、合成用画像を用意
        pil_rgba_bg_temp = Image.new('RGBA', pil_rgba_background_image.size,
                                    (255, 255, 255, 0))
        # 座標を指定し重ね合わせる
        pil_rgba_bg_temp.paste(pil_rgba_overlay_image, point, pil_rgba_overlay_image)
        result_image = Image.alpha_composite(pil_rgba_background_image, pil_rgba_bg_temp)

        # OpenCV形式画像へ変換
        cv_result_image = cv2.cvtColor(np.asarray(result_image), cv2.COLOR_RGBA2BGRA)

        return cv_result_image

    # カメラ映像取得
    cap = cv2.VideoCapture(DEVICE_ID)

    # 初期フレームの読込
    end_flag, c_frame = cap.read()
    height, width, channels = c_frame.shape

    # ウィンドウの準備
    cv2.namedWindow(GAUSSIAN_WINDOW_NAME)

    # 変換処理ループ
    while end_flag == True:

        # 画像の取得と顔の検出
        img = c_frame
        face_list = cascade.detectMultiScale(img, minSize=(100, 100))

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.resize(img_gray, (64, 64))
        img_array = img_to_array(img_gray)
        pImg = np.expand_dims(img_array, axis=0) / 255

        # モデルに投げる
        pre = for_model(pImg)
        print(pre)

        # confidenceが一番高いものが"sad"か"fear"かつ、confidenceが0.3以上だったら表示する
        if len(pre) != 0:
            if (pre[0][0] == "sad" or pre[0][0] == "fear") and pre[0][1] > 0.3:
                for (x, y, w, h) in face_list:
                    # color = (0, 0, 225)
                    # pen_w = 3
                    # cv2.rectangle(img, (x, y), (x+w, y+h), color, thickness = pen_w)
                    size = (w, h)
                    point = (x, y)
                    resize_smile = cv2.resize(smile, size)
                    img = overlay(img, resize_smile, point)
                    cv2.putText(img, "Let's smile!", (x,y-30), cv2.FONT_HERSHEY_DUPLEX | cv2.FONT_ITALIC, 2.5, (100,100,200), 4, cv2.LINE_AA)
                    # smile_flag = True

            elif pre[0][0] == "happy" and pre[0][1] > 0.3:
                for (x, y, w, h) in face_list:
                    # color = (255, 0, 0)
                    # pen_w = 3
                    # cv2.rectangle(img, (x, y), (x+w, y+h), color, thickness = pen_w)
                    size = (w, h)
                    point = (x, y)
                    cv2.putText(img, "Good smile!", (x,y-30), cv2.FONT_HERSHEY_DUPLEX | cv2.FONT_ITALIC, 2.5, (46,204,250), 4, cv2.LINE_AA)


        # フレーム表示
        cv2.imshow(GAUSSIAN_WINDOW_NAME, img)

        # if smile_flag:
        #     #音を鳴らすコード
        #     pygame.mixer.music.play()
        #     time.sleep(1)
        #     pygame.mixer.music.stop()
        #     smile_flag = False


        # Escキーで終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # 次のフレーム読み込み
        end_flag, c_frame = cap.read()

    # 終了処理
    cv2.destroyAllWindows()
    cap.release()
