import cv2

# カメラの準備 (0番目のカメラ)
cap = cv2.VideoCapture(0)

# 1枚だけ画像を読み込む
ret, frame = cap.read()

if ret:
    # 成功したら 'test.jpg' という名前で保存
    cv2.imwrite('test.jpg', frame)
    print("カメラの撮影に成功しました！test.jpgを保存しました。")
else:
    print("エラー：カメラを認識できませんでした。")

# カメラを終了する
cap.release()